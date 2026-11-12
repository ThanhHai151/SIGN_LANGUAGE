import fs from 'fs';
import readline from 'readline';
import { evaluateTrialFrames, buildEvaluationReport } from '../src/components/pilotProtocol.js';

async function parseJsonl(filePath) {
  const fileStream = fs.createReadStream(filePath);
  const rl = readline.createInterface({ input: fileStream, crlfDelay: Infinity });
  const frames = [];
  for await (const line of rl) {
    if (line.trim()) {
      frames.push(JSON.parse(line));
    }
  }
  return frames;
}

async function run() {
  console.log("=== COMPARING H & N BEFORE vs AFTER SCORING REPAIR ===");

  const hSummaryOld = JSON.parse(fs.readFileSync('backend/data/diagnostics/pilot_eval_29_pilot_20261112_041651_86b712_20261112_041820.json', 'utf8'));
  const nSummaryOld = JSON.parse(fs.readFileSync('backend/data/diagnostics/pilot_eval_29_pilot_20261112_040819_b6dc69_20261112_041644.json', 'utf8'));

  const hFrames = await parseJsonl('backend/data/diagnostics/pilot_eval_29_pilot_20261112_041651_86b712_20261112_041820_frames.jsonl');
  const nFrames = await parseJsonl('backend/data/diagnostics/pilot_eval_29_pilot_20261112_040819_b6dc69_20261112_041644_frames.jsonl');

  // Group by rep using elapsed_sec reset
  function groupByRep(frames) {
    const reps = {};
    let currentRep = 1;
    let prevSec = -1;
    frames.forEach(f => {
      const sec = f.elapsed_sec || 0;
      if (sec < prevSec) {
        currentRep += 1;
      }
      if (!reps[currentRep]) reps[currentRep] = [];
      reps[currentRep].push(f);
      prevSec = sec;
    });
    return reps;
  }

  // Evaluate H
  const hReps = groupByRep(hFrames);
  const hNewTrials = [];
  console.log("\n--- LETTER H RE-EVALUATION BY REP ---");
  console.log("Rep | Old Match% | Old Status | New Frames | Match | NEUTRAL | Mismatch | New Match% | New Status");
  console.log("----+------------+------------+------------+-------+---------+----------+------------+-----------");
  const oldHTrials = hSummaryOld.letters?.H?.trials || [];

  for (let r = 1; r <= 10; r++) {
    const repFrames = hReps[r] || [];
    const oldT = oldHTrials.find(t => t.rep === r) || {};
    const evalRes = evaluateTrialFrames(repFrames, 'H');
    evalRes.rep = r;
    hNewTrials.push(evalRes);
    console.log(
      `${String(r).padStart(3)} | ` +
      `${String(oldT.matchPct ?? 'N/A').padStart(10)}% | ` +
      `${String(oldT.status ?? 'N/A').padStart(10)} | ` +
      `${String(evalRes.totalFrames).padStart(10)} | ` +
      `${String(evalRes.matchFrames).padStart(5)} | ` +
      `${String(evalRes.neutralFrames).padStart(7)} | ` +
      `${String(evalRes.mismatchFrames).padStart(8)} | ` +
      `${String(evalRes.matchPct).padStart(9)}% | ` +
      `${evalRes.status}`
    );
  }

  const hNewReport = buildEvaluationReport('pilot_h', { 'H': hNewTrials }, { trialsPerLetter: 10 });
  console.log("\nSummary Letter H:");
  console.log(`Old: status=${hSummaryOld.letters?.H?.status}, accuracy=${hSummaryOld.letters?.H?.accuracyPct}%, passedTrials=${hSummaryOld.letters?.H?.passedTrials}/10`);
  console.log(`New: status=${hNewReport.letters.H.status}, accuracy=${hNewReport.letters.H.accuracyPct}%, passedTrials=${hNewReport.letters.H.passedTrials}/10`);

  // Evaluate N
  const nReps = groupByRep(nFrames);
  const nNewTrials = [];
  console.log("\n--- LETTER N RE-EVALUATION BY REP ---");
  console.log("Rep | Old Match% | Old Status | New Frames | Match | NEUTRAL | Mismatch | New Match% | New Status");
  console.log("----+------------+------------+------------+-------+---------+----------+------------+-----------");
  const oldNTrials = nSummaryOld.letters?.N?.trials || [];

  for (let r = 1; r <= 10; r++) {
    const repFrames = nReps[r] || [];
    const oldT = oldNTrials.find(t => t.rep === r) || {};
    const evalRes = evaluateTrialFrames(repFrames, 'N');
    evalRes.rep = r;
    nNewTrials.push(evalRes);
    console.log(
      `${String(r).padStart(3)} | ` +
      `${String(oldT.matchPct ?? 'N/A').padStart(10)}% | ` +
      `${String(oldT.status ?? 'N/A').padStart(10)} | ` +
      `${String(evalRes.totalFrames).padStart(10)} | ` +
      `${String(evalRes.matchFrames).padStart(5)} | ` +
      `${String(evalRes.neutralFrames).padStart(7)} | ` +
      `${String(evalRes.mismatchFrames).padStart(8)} | ` +
      `${String(evalRes.matchPct).padStart(9)}% | ` +
      `${evalRes.status}`
    );
  }

  const nNewReport = buildEvaluationReport('pilot_n', { 'N': nNewTrials }, { trialsPerLetter: 10 });
  console.log("\nSummary Letter N:");
  console.log(`Old: status=${nSummaryOld.letters?.N?.status}, accuracy=${nSummaryOld.letters?.N?.accuracyPct}%, passedTrials=${nSummaryOld.letters?.N?.passedTrials}/10`);
  console.log(`New: status=${nNewReport.letters.N.status}, accuracy=${nNewReport.letters.N.accuracyPct}%, passedTrials=${nNewReport.letters.N.passedTrials}/10`);
}

run().catch(console.error);
