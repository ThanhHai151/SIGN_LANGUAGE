import GUI from 'lil-gui';
import {
  getMixamoPoseConfig,
  getPoseForSymbol,
  getSavedPoseOverrides,
  resolveVietnameseSign,
  savePoseForLetter,
  VIETNAMESE_SYMBOL_OPTIONS
} from './MixamoPoseAdapter';

const FINGERS = {
  Thumb: ['RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightHandThumb4'],
  Index: ['RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3'],
  Middle: ['RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3'],
  Ring: ['RightHandRing1', 'RightHandRing2', 'RightHandRing3'],
  Pinky: ['RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3']
};

const EDITABLE_BONES = ['RightHand', ...Object.values(FINGERS).flat()];

function findBone(bonesMap, name) {
  if (!bonesMap || typeof bonesMap.get !== 'function') return null;
  const baseName = name.replace(/^mixamorig/, '');
  return bonesMap.get(baseName) ?? bonesMap.get(`mixamorig${baseName}`) ?? null;
}

function readRigPose(bonesMap, basePose) {
  const current = new Map(basePose.map((target) => [target.bone, target]));
  for (const boneName of EDITABLE_BONES) {
    const rotation = findBone(bonesMap, boneName)?.rotation;
    if (!rotation) continue;
    current.set(boneName, {
      ...(current.get(boneName) || { bone: boneName }),
      bone: boneName,
      rot: [rotation.x, rotation.y, rotation.z]
    });
  }
  return [...current.values()];
}

export function setupHandGUI(bonesMap, {
  symbol = 'A',
  onPoseChange = () => {},
  onPreviewSequence = () => {}
} = {}) {
  const gui = new GUI({ title: 'Hand Pose Tuner' });
  const selection = { symbol: Object.values(VIETNAMESE_SYMBOL_OPTIONS).includes(symbol) ? symbol : 'A' };
  let suppressSelectionApply = false;

  const applySelectedPose = (selectedSymbol) => {
    selection.symbol = selectedSymbol;
    const pose = getPoseForSymbol(selectedSymbol);
    for (const target of pose) {
      if (!EDITABLE_BONES.includes(target.bone)) continue;
      const bone = findBone(bonesMap, target.bone);
      if (bone?.rotation && Array.isArray(target.rot)) bone.rotation.set(...target.rot);
    }
    onPoseChange(pose);
  };

  const selectionFolder = gui.addFolder('Letter / Symbol Select');
  const symbolController = selectionFolder
    .add(selection, 'symbol', VIETNAMESE_SYMBOL_OPTIONS)
    .name('Letter / Symbol');
  symbolController.onChange((nextSymbol) => {
    if (!suppressSelectionApply) applySelectedPose(nextSymbol);
  });
  gui.setLetter = (nextSymbol, { apply = true } = {}) => {
    if (Object.values(VIETNAMESE_SYMBOL_OPTIONS).includes(nextSymbol)) {
      suppressSelectionApply = !apply;
      symbolController.setValue(nextSymbol);
      suppressSelectionApply = false;
      if (apply) applySelectedPose(nextSymbol);
    }
  };
  gui.setSymbol = gui.setLetter;

  const handFolder = gui.addFolder('Right Hand & Fingers');
  const wrist = findBone(bonesMap, 'RightHand');
  if (wrist?.rotation) {
    const folder = handFolder.addFolder('RightHand · Wrist');
    for (const axis of ['x', 'y', 'z']) {
      folder.add(wrist.rotation, axis, -Math.PI, Math.PI, 0.01)
        .name(`Rotation ${axis.toUpperCase()}`)
        .listen()
        .onChange(() => onPoseChange(readRigPose(bonesMap, getPoseForSymbol(selection.symbol))));
    }
  }

  for (const [fingerName, boneNames] of Object.entries(FINGERS)) {
    const fingerFolder = handFolder.addFolder(fingerName);
    for (const boneName of boneNames) {
      const bone = findBone(bonesMap, boneName);
      if (!bone?.rotation) continue;
      const jointFolder = fingerFolder.addFolder(boneName);
      for (const axis of ['x', 'y', 'z']) {
        jointFolder.add(bone.rotation, axis, -Math.PI, Math.PI, 0.01)
          .name(`Rotation ${axis.toUpperCase()}`)
          .listen()
          .onChange(() => onPoseChange(readRigPose(bonesMap, getPoseForSymbol(selection.symbol))));
      }
    }
  }

  const actions = {
    'Preview Sequence': () => onPreviewSequence(resolveVietnameseSign(selection.symbol), selection.symbol),
    'Reset Hand': () => {
      for (const boneName of EDITABLE_BONES) findBone(bonesMap, boneName)?.rotation?.set(0, 0, 0);
      onPoseChange(readRigPose(bonesMap, getPoseForSymbol(selection.symbol)));
    },
    'Copy Pose Data': async () => {
      const pose = readRigPose(bonesMap, getPoseForSymbol(selection.symbol))
        .filter((target) => EDITABLE_BONES.includes(target.bone))
        .map(({ bone, rot }) => ({ bone, rot: rot.map((angle) => Number(angle.toFixed(3))) }));
      const json = JSON.stringify({ symbol: selection.symbol, targets: pose }, null, 2);
      console.log(json);
      try {
        await navigator.clipboard.writeText(json);
      } catch (error) {
        console.warn('Could not copy pose data to clipboard.', error);
      }
    },
    'Save Pose for Symbol': () => {
      const pose = readRigPose(bonesMap, getPoseForSymbol(selection.symbol));
      if (savePoseForLetter(selection.symbol, pose)) {
        console.info(`Saved tuned pose for ${selection.symbol}.`);
      } else {
        console.warn(`This symbol cannot be saved as a static pose: ${selection.symbol}.`);
      }
    },
    'Export Config JSON': async () => {
      const json = JSON.stringify({
        ...getMixamoPoseConfig(),
        savedOverrides: getSavedPoseOverrides()
      }, null, 2);
      const blob = new Blob([json], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = 'mixamo-vietnamese-fingerspelling.json';
      anchor.click();
      URL.revokeObjectURL(url);
      try {
        await navigator.clipboard.writeText(json);
      } catch {
        // File download remains available if clipboard permissions are denied.
      }
    }
  };

  const actionsFolder = gui.addFolder('Actions');
  actionsFolder.add(actions, 'Preview Sequence');
  actionsFolder.add(actions, 'Reset Hand');
  actionsFolder.add(actions, 'Copy Pose Data');
  actionsFolder.add(actions, 'Save Pose for Symbol');
  actionsFolder.add(actions, 'Export Config JSON');

  applySelectedPose(selection.symbol);
  return gui;
}
