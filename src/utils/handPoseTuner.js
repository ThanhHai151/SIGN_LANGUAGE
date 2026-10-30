import GUI from 'lil-gui';
import { getPoseForLetter, savePoseForLetter } from './MixamoPoseAdapter';

const FINGERS = {
  Thumb: ['RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightHandThumb4'],
  Index: ['RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3'],
  Middle: ['RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3'],
  Ring: ['RightHandRing1', 'RightHandRing2', 'RightHandRing3'],
  Pinky: ['RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3']
};

function findBone(bonesMap, name) {
  if (!bonesMap || typeof bonesMap.get !== 'function') return null;

  const baseName = name.replace(/^mixamorig/, '');
  return bonesMap.get(baseName) ?? bonesMap.get(`mixamorig${baseName}`) ?? null;
}

export function setupHandGUI(bonesMap, { letter = 'A', onPoseChange = () => {} } = {}) {
  const gui = new GUI({ title: 'Hand Pose Tuner' });
  const fingerBones = Object.values(FINGERS).flat();
  const selection = { letter: /^[A-Z]$/i.test(letter) ? letter.toUpperCase() : 'A' };

  const loadSelectedPose = (selectedLetter) => {
    selection.letter = selectedLetter;
    const pose = getPoseForLetter(selectedLetter);
    for (const target of pose) {
      if (!/^RightHand(?:Thumb|Index|Middle|Ring|Pinky)\d+$/.test(target.bone)) continue;
      findBone(bonesMap, target.bone)?.rotation?.set(...target.rot);
    }
    onPoseChange(pose);
  };

  const poseFolder = gui.addFolder('Sign Pose');
  const letterController = poseFolder
    .add(selection, 'letter', [...'ABCDEFGHIJKLMNOPQRSTUVWXYZ'])
    .name('Letter');
  letterController.onChange(loadSelectedPose);
  gui.setLetter = (selectedLetter) => {
    const normalizedLetter = typeof selectedLetter === 'string' ? selectedLetter.trim().toUpperCase() : '';
    if (/^[A-Z]$/.test(normalizedLetter)) letterController.setValue(normalizedLetter);
  };

  for (const [fingerName, boneNames] of Object.entries(FINGERS)) {
    const folder = gui.addFolder(fingerName);

    for (const boneName of boneNames) {
      const bone = findBone(bonesMap, boneName);
      if (!bone?.rotation) continue;

      const jointFolder = folder.addFolder(boneName);
      for (const axis of ['x', 'y', 'z']) {
        jointFolder.add(bone.rotation, axis, -Math.PI, Math.PI, 0.01).listen();
      }
    }
  }

  const actions = {
    'Reset Hand': () => {
      for (const boneName of fingerBones) {
        findBone(bonesMap, boneName)?.rotation?.set(0, 0, 0);
      }
    },
    'Copy Pose Data': async () => {
      const pose = fingerBones.flatMap((boneName) => {
        const rotation = findBone(bonesMap, boneName)?.rotation;
        if (!rotation) return [];

        return [{
          bone: boneName,
          rot: [rotation.x, rotation.y, rotation.z].map((angle) => Number(angle.toFixed(2)))
        }];
      });
      const json = JSON.stringify(pose, null, 2);

      console.log(json);
      try {
        await navigator.clipboard.writeText(json);
      } catch (error) {
        console.warn('Could not copy hand pose data to clipboard.', error);
      }
    },
    'Save Pose for Letter': () => {
      const pose = fingerBones.flatMap((boneName) => {
        const rotation = findBone(bonesMap, boneName)?.rotation;
        if (!rotation) return [];
        return [{
          bone: boneName,
          rot: [rotation.x, rotation.y, rotation.z].map((angle) => Number(angle.toFixed(2)))
        }];
      });

      if (savePoseForLetter(selection.letter, pose)) {
        console.info(`Saved tuned hand pose for ${selection.letter}.`);
      } else {
        console.warn('Could not save the tuned hand pose in browser storage.');
      }
    }
  };

  const actionsFolder = gui.addFolder('Actions');
  actionsFolder.add(actions, 'Reset Hand');
  actionsFolder.add(actions, 'Copy Pose Data');
  actionsFolder.add(actions, 'Save Pose for Letter');

  loadSelectedPose(selection.letter);
  return gui;
}
