// verify_hud_cases.cjs - Verification script for Task 12.1.10
// Tests all 9 required verification cases

const STATIC_LETTERS_22 = [
  'A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L',
  'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y'
];
const DEFAULT_REPS_PER_LETTER = 10;
const TRIAL_DURATION_SEC = 5.0;

// Simulation engine mirroring PilotTestingHUD state machine
class PilotHUDModel {
  constructor() {
    this.sessionId = 'pilot_20261108_0310';
    this.campaignMode = 'full'; // 'full' | 'selective'
    this.selectedLetters = [...STATIC_LETTERS_22];
    this.customRepsPerLetter = DEFAULT_REPS_PER_LETTER;
    this.isCampaignLocked = false;
    this.selectedLetter = 'A';
    this.currentRep = 1;
    this.currentAttempt = 1;
    this.trialState = 'idle'; // 'idle' | 'recording' | 'paused' | 'pending_flush' | 'flushed' | 'flush_error'
    this.progressMatrix = {};
    STATIC_LETTERS_22.forEach((l) => {
      this.progressMatrix[l] = { completedReps: [], attempts: {} };
    });
    this.alertMessages = [];
  }

  alert(msg) {
    this.alertMessages.push(msg);
  }

  getActiveLetters() {
    if (this.campaignMode === 'full') return STATIC_LETTERS_22;
    return STATIC_LETTERS_22.filter(l => this.selectedLetters.includes(l));
  }

  getActiveReps() {
    if (this.campaignMode === 'full') return DEFAULT_REPS_PER_LETTER;
    const parsed = parseInt(this.customRepsPerLetter, 10);
    return isNaN(parsed) || parsed < 1 ? 1 : Math.min(parsed, 50);
  }

  getTotalCampaignTrials() {
    return this.getActiveLetters().length * this.getActiveReps();
  }

  getTrialId() {
    const padRep = String(this.currentRep).padStart(2, '0');
    const padAtt = String(this.currentAttempt).padStart(2, '0');
    return `${this.sessionId}_${this.selectedLetter}_rep${padRep}_att${padAtt}`;
  }

  getTotalCompletedTrials() {
    const activeLetters = this.getActiveLetters();
    const activeReps = this.getActiveReps();
    return activeLetters.reduce((sum, letter) => {
      const item = this.progressMatrix[letter];
      if (!item || !item.completedReps) return sum;
      const validCompleted = item.completedReps.filter(r => r <= activeReps);
      return sum + validCompleted.length;
    }, 0);
  }

  getOverallProgressPct() {
    const total = this.getTotalCampaignTrials();
    if (total === 0) return 0;
    return Math.min(100, (this.getTotalCompletedTrials() / total) * 100);
  }

  getCurrentGlobalTrialNum() {
    const activeLetters = this.getActiveLetters();
    const activeReps = this.getActiveReps();
    const idx = Math.max(0, activeLetters.indexOf(this.selectedLetter));
    return (idx * activeReps) + Math.min(this.currentRep, activeReps);
  }

  syncSelectedLetter() {
    const activeLetters = this.getActiveLetters();
    if (activeLetters.length > 0 && !activeLetters.includes(this.selectedLetter)) {
      this.selectedLetter = activeLetters[0];
      this.currentRep = 1;
      this.currentAttempt = 1;
    }
  }

  startTrial(cameraActive = true) {
    const activeLetters = this.getActiveLetters();
    if (activeLetters.length === 0) {
      this.alert('⚠️ Chưa có chữ cái nào được chọn!');
      return false;
    }
    if (!cameraActive) {
      this.alert('Vui lòng bật Camera!');
      return false;
    }
    this.isCampaignLocked = true;
    this.trialState = 'recording';
    return true;
  }

  pauseTrial() {
    if (this.trialState !== 'recording') return;
    this.trialState = 'paused';
  }

  resumeTrial() {
    if (this.trialState !== 'paused') return;
    this.trialState = 'recording';
  }

  executeFlush(success = true) {
    this.trialState = 'pending_flush';
    if (success) {
      const letterData = this.progressMatrix[this.selectedLetter];
      if (!letterData.completedReps.includes(this.currentRep)) {
        letterData.completedReps.push(this.currentRep);
      }
      letterData.attempts[this.currentRep] = this.currentAttempt;
      this.trialState = 'flushed';
    } else {
      this.trialState = 'flush_error';
    }
  }

  nextTrial() {
    if (this.trialState !== 'flushed') return;
    const activeReps = this.getActiveReps();
    if (this.currentRep < activeReps) {
      this.currentRep += 1;
      this.currentAttempt = 1;
      this.trialState = 'idle';
    } else {
      this.nextLetter(false);
    }
  }

  nextLetter(forceConfirm = true) {
    const activeLetters = this.getActiveLetters();
    const currIdx = activeLetters.indexOf(this.selectedLetter);
    const nextIdx = currIdx + 1;
    if (nextIdx >= activeLetters.length) {
      this.alert(`🎉 Đã hoàn thành toàn bộ chiến dịch kiểm thử (${this.getTotalCampaignTrials()}/${this.getTotalCampaignTrials()} lượt)!`);
      return;
    }
    this.selectedLetter = activeLetters[nextIdx];
    this.currentRep = 1;
    this.currentAttempt = 1;
    this.trialState = 'idle';
  }

  promptRetest() {
    this.currentAttempt += 1;
    this.trialState = 'idle';
  }
}

console.log('================================================================');
console.log('VERIFYING 9 TASK 12.1.10 TEST CASES');
console.log('================================================================');

let passedCases = 0;

// CASE 1: Select only H, 2 reps -> total 2 trials
{
  const hud = new PilotHUDModel();
  hud.campaignMode = 'selective';
  hud.selectedLetters = ['H'];
  hud.customRepsPerLetter = 2;
  hud.syncSelectedLetter();

  const activeLetters = hud.getActiveLetters();
  const totalTrials = hud.getTotalCampaignTrials();
  const initialTrialNum = hud.getCurrentGlobalTrialNum();
  const initialTrialId = hud.getTrialId();

  const ok = (activeLetters.length === 1 && activeLetters[0] === 'H') &&
             (totalTrials === 2) &&
             (hud.selectedLetter === 'H') &&
             (initialTrialNum === 1) &&
             (initialTrialId === 'pilot_20261108_0310_H_rep01_att01');

  if (ok) {
    console.log('✅ CASE 1 PASSED: Select only H, 2 reps -> total 2 trials (trialId: ' + initialTrialId + ')');
    passedCases++;
  } else {
    console.error('❌ CASE 1 FAILED');
  }
}

// CASE 2: Select H, K, V, 3 reps -> total 9 trials (0/9 to 9/9)
{
  const hud = new PilotHUDModel();
  hud.campaignMode = 'selective';
  hud.selectedLetters = ['H', 'K', 'V'];
  hud.customRepsPerLetter = 3;
  hud.syncSelectedLetter();

  const totalTrials = hud.getTotalCampaignTrials();
  let trialSequence = [];

  for (let l = 0; l < 3; l++) {
    for (let r = 1; r <= 3; r++) {
      const trialNum = hud.getCurrentGlobalTrialNum();
      const trialId = hud.getTrialId();
      trialSequence.push({ letter: hud.selectedLetter, rep: hud.currentRep, trialNum, trialId });
      hud.startTrial();
      hud.executeFlush(true);
      hud.nextTrial();
    }
  }

  const completed = hud.getTotalCompletedTrials();
  const pct = hud.getOverallProgressPct();
  const alertOccurred = hud.alertMessages.some(m => m.includes('Đã hoàn thành toàn bộ chiến dịch'));

  const ok = (totalTrials === 9) &&
             (completed === 9) &&
             (pct === 100) &&
             (trialSequence.length === 9) &&
             (trialSequence[0].trialNum === 1 && trialSequence[0].trialId === 'pilot_20261108_0310_H_rep01_att01') &&
             (trialSequence[3].trialNum === 4 && trialSequence[3].trialId === 'pilot_20261108_0310_K_rep01_att01') &&
             (trialSequence[6].trialNum === 7 && trialSequence[6].trialId === 'pilot_20261108_0310_V_rep01_att01') &&
             (trialSequence[8].trialNum === 9 && trialSequence[8].trialId === 'pilot_20261108_0310_V_rep03_att01') &&
             alertOccurred;

  if (ok) {
    console.log('✅ CASE 2 PASSED: Select H, K, V, 3 reps -> 9 trials (1/9 to 9/9), exact letter order and completion alert');
    passedCases++;
  } else {
    console.error('❌ CASE 2 FAILED', { totalTrials, completed, pct, alertOccurred });
  }
}

// CASE 3: Select A, M, N, X, 2 reps -> total 8 trials
{
  const hud = new PilotHUDModel();
  hud.campaignMode = 'selective';
  hud.selectedLetters = ['A', 'M', 'N', 'X'];
  hud.customRepsPerLetter = 2;
  hud.syncSelectedLetter();

  const activeLetters = hud.getActiveLetters();
  const totalTrials = hud.getTotalCampaignTrials();

  const ok = (activeLetters.join(',') === 'A,M,N,X') && (totalTrials === 8);

  if (ok) {
    console.log('✅ CASE 3 PASSED: Select A, M, N, X, 2 reps -> total 8 trials (activeLetters: ' + activeLetters.join(', ') + ')');
    passedCases++;
  } else {
    console.error('❌ CASE 3 FAILED');
  }
}

// CASE 4: Select all 22 letters in Selective Testing, 10 reps -> 220 trials (equivalent to Mode A)
{
  const hudA = new PilotHUDModel();
  hudA.campaignMode = 'full';

  const hudB = new PilotHUDModel();
  hudB.campaignMode = 'selective';
  hudB.selectedLetters = [...STATIC_LETTERS_22];
  hudB.customRepsPerLetter = 10;
  hudB.syncSelectedLetter();

  const ok = (hudA.getTotalCampaignTrials() === 220) &&
             (hudB.getTotalCampaignTrials() === 220) &&
             (hudA.getActiveLetters().length === hudB.getActiveLetters().length) &&
             (hudA.getActiveReps() === hudB.getActiveReps());

  if (ok) {
    console.log('✅ CASE 4 PASSED: 22 letters in Selective Mode with 10 reps matches Mode A exactly (220 trials)');
    passedCases++;
  } else {
    console.error('❌ CASE 4 FAILED');
  }
}

// CASE 5: 0 letters selected -> cannot start, warning shown
{
  const hud = new PilotHUDModel();
  hud.campaignMode = 'selective';
  hud.selectedLetters = [];
  hud.syncSelectedLetter();

  const started = hud.startTrial(true);
  const ok = (!started) &&
             (hud.trialState === 'idle') &&
             (hud.alertMessages.some(m => m.includes('Chưa có chữ cái nào')));

  if (ok) {
    console.log('✅ CASE 5 PASSED: 0 letters selected -> startTrial blocked and warning alert triggered');
    passedCases++;
  } else {
    console.error('❌ CASE 5 FAILED', { started, state: hud.trialState, alerts: hud.alertMessages });
  }
}

// CASE 6: Retest trial -> attempt increments (att02), previous data preserved
{
  const hud = new PilotHUDModel();
  hud.campaignMode = 'selective';
  hud.selectedLetters = ['H'];
  hud.customRepsPerLetter = 1;
  hud.syncSelectedLetter();

  hud.startTrial();
  hud.executeFlush(true);
  const trialIdAtt01 = hud.getTrialId();

  // Tester prompts Retest
  hud.promptRetest();
  const trialIdAtt02 = hud.getTrialId();

  const ok = (trialIdAtt01 === 'pilot_20261108_0310_H_rep01_att01') &&
             (trialIdAtt02 === 'pilot_20261108_0310_H_rep01_att02') &&
             (hud.currentAttempt === 2) &&
             (hud.trialState === 'idle') &&
             (hud.progressMatrix['H'].attempts[1] === 1);

  if (ok) {
    console.log('✅ CASE 6 PASSED: Retest increments attempt (att01 -> att02), prior attempt preserved');
    passedCases++;
  } else {
    console.error('❌ CASE 6 FAILED', { trialIdAtt01, trialIdAtt02 });
  }
}

// CASE 7: Pause / Resume -> trial state transitions correctly
{
  const hud = new PilotHUDModel();
  hud.campaignMode = 'selective';
  hud.selectedLetters = ['H'];
  hud.syncSelectedLetter();

  hud.startTrial();
  const recState = hud.trialState;
  hud.pauseTrial();
  const pauseState = hud.trialState;
  hud.resumeTrial();
  const resumeState = hud.trialState;

  const ok = (recState === 'recording') && (pauseState === 'paused') && (resumeState === 'recording');

  if (ok) {
    console.log('✅ CASE 7 PASSED: Pause / Resume preserves trial lifecycle (recording -> paused -> recording)');
    passedCases++;
  } else {
    console.error('❌ CASE 7 FAILED', { recState, pauseState, resumeState });
  }
}

// CASE 8: Flush error -> hard gate blocks progression
{
  const hud = new PilotHUDModel();
  hud.campaignMode = 'selective';
  hud.selectedLetters = ['H'];
  hud.customRepsPerLetter = 2;
  hud.syncSelectedLetter();

  hud.startTrial();
  hud.executeFlush(false); // flush fails!
  const stateAfterFail = hud.trialState;

  // Try to advance to next trial
  hud.nextTrial(); // should be blocked!
  const repAfterBlockedNext = hud.currentRep;

  const ok = (stateAfterFail === 'flush_error') && (repAfterBlockedNext === 1);

  if (ok) {
    console.log('✅ CASE 8 PASSED: Flush failure enters flush_error and blocks nextTrial advancement');
    passedCases++;
  } else {
    console.error('❌ CASE 8 FAILED', { stateAfterFail, repAfterBlockedNext });
  }
}

// CASE 9: Mode A regression check (full campaign integrity)
{
  const hud = new PilotHUDModel();
  hud.campaignMode = 'full';
  const totalLetters = hud.getActiveLetters().length;
  const reps = hud.getActiveReps();
  const total = hud.getTotalCampaignTrials();

  const ok = (totalLetters === 22) && (reps === 10) && (total === 220) && (hud.selectedLetter === 'A');

  if (ok) {
    console.log('✅ CASE 9 PASSED: Mode A regression confirmed (22 letters x 10 reps = 220 trials, starts at A)');
    passedCases++;
  } else {
    console.error('❌ CASE 9 FAILED');
  }
}

console.log('================================================================');
console.log(`SUMMARY: ${passedCases}/9 CASES PASSED`);
console.log('================================================================');

