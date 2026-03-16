// lib/nnue_training.dart
import 'dart:math' show Random, pi, cos, pow;
import 'package:dart_fish/chess_nnue3.dart'; // ChessWithNNUE
import 'package:dart_fish/alphabeta_train2.dart'; // NNUESearcher, NNUETrainer (your implementations)
import 'package:dart_fish/nnue_logic_batch2.dart'; // NNUE, TrainingPosition
import 'package:dart_fish/nnue_persistence2.dart';

import 'chess3.dart'; // NNUESerializer

// -----------------------------
// Utilities: tiny args parser
// -----------------------------
class _Args {
  final Map<String, String> _map;
  _Args(this._map);

  static _Args parse(List<String> args) {
    final m = <String, String>{};
    for (final a in args) {
      final eq = a.indexOf('=');
      if (a.startsWith('--') && eq > 2) {
        final k = a.substring(2, eq).trim();
        final v = a.substring(eq + 1).trim();
        if (k.isNotEmpty) m[k] = v;
      }
    }
    return _Args(m);
  }

  String getString(String key, String def) => _map[key] ?? def;
  int getInt(String key, int def) {
    final v = _map[key];
    if (v == null) return def;
    return int.tryParse(v) ?? def;
  }

  double getDouble(String key, double def) {
    final v = _map[key];
    if (v == null) return def;
    return double.tryParse(v) ?? def;
  }

  bool getBool(String key, bool def) {
    final v = _map[key];
    if (v == null) return def;
    final low = v.toLowerCase();
    if (low == 'true' || low == '1' || low == 'yes') return true;
    if (low == 'false' || low == '0' || low == 'no') return false;
    return def;
  }
}

// -----------------------------
// LR Schedules
// -----------------------------
enum LrSchedule { fixed, cosine, step }

LrSchedule _parseSchedule(String s) {
  switch (s.toLowerCase()) {
    case 'cosine':
      return LrSchedule.cosine;
    case 'step':
      return LrSchedule.step;
    case 'fixed':
    default:
      return LrSchedule.fixed;
  }
}

double _cosineDecay(double baseLr, int t, int T) {
  // LR_t = 0.5 * base * (1 + cos(pi * t / T))
  return 0.5 * baseLr * (1.0 + cos(pi * (t / (T <= 0 ? 1 : T))));
}

double _stepDecay(double baseLr, int t, int stepEvery, double gamma) {
  // LR_t = base * gamma^(floor(t / stepEvery))
  if (stepEvery <= 0) return baseLr;
  final k = (t ~/ stepEvery);
  return baseLr * pow(gamma, k).toDouble();
}

// -----------------------------
// Helpers shared by both modes
// -----------------------------
List<Piece?> _cloneBoard(List<Piece?> board) {
  final out = List<Piece?>.filled(board.length, null);
  for (int i = 0; i < board.length; i++) {
    final p = board[i];
    if (p != null) out[i] = Piece(p.type, p.color);
  }
  return out;
}

List<TrainingPosition> _buildOverfitBatch(
  ChessWithNNUE game,
  String fen,
  double targetCp,
  int batchSize,
) {
  game.load(fen); // ensures accumulator sync
  final boardCopy = _cloneBoard(game.board);
  final stm = game.turn;
  return List.generate(
    batchSize,
    (_) => TrainingPosition(boardCopy, stm, targetCp),
  );
}

double _predictCp(ChessWithNNUE game, String fen) {
  game.load(fen);
  return game.nnueEvaluation;
}

Future<void> _tryLoad(NNUE nnue, String path) async {
  try {
    await NNUESerializer.load(nnue, path);
    print("✅ Loaded NNUE model from $path");
  } catch (e) {
    print(
      "⚠️ No weights file found or load failed at $path. Starting with random weights. ($e)",
    );
  }
}

Future<void> _trySave(NNUE nnue, String path) async {
  try {
    await NNUESerializer.save(nnue, path);
    print("💾 Saved NNUE model to $path");
  } catch (e) {
    print("⚠️ Save failed: $e");
  }
}

// -----------------------------
// General Training Iterations
// -----------------------------
Future<void> runTrainIterations({
  required ChessWithNNUE game,
  required NNUETrainer trainer,
  required NNUESearcher searcher,
  required int iterations,
  int samplesPerIter = 64,
  int labelDepth = 2,
  int evalDepth = 3,
  double baseLr = 1e-3,
  LrSchedule schedule = LrSchedule.fixed,
  int stepEvery = 25,
  double gamma = 0.5,
  bool autosave = true,
  String savePath = 'chess_model_iter.json',
  int saveEvery = 10,
}) async {
  print(
    '--- NNUE Training (iters=$iterations, samples=$samplesPerIter, labelDepth=$labelDepth, evalDepth=$evalDepth) ---',
  );

  for (int it = 1; it <= iterations; it++) {
    // Select LR
    double lr;
    switch (schedule) {
      case LrSchedule.cosine:
        lr = _cosineDecay(baseLr, it - 1, iterations);
        break;
      case LrSchedule.step:
        lr = _stepDecay(baseLr, it - 1, stepEvery, gamma);
        break;
      case LrSchedule.fixed:
      default:
        lr = baseLr;
        break;
    }

    print('\n--- Iteration $it/$iterations (lr=${lr.toStringAsFixed(6)}) ---');

    // Label & collect positions
    trainer.generateData(samplesPerIter, labelDepth);

    // Train
    trainer.runEpoch(lr);

    // Evaluate baseline from start pos
    game.reset();
    final res = searcher.search(evalDepth);
    print(
      'Eval @ depth $evalDepth -> ${res.score.toStringAsFixed(2)} cp | nodes: ${res.nodes}',
    );

    // Autosave
    if (autosave && (it % saveEvery == 0 || it == iterations)) {
      await _trySave(game.nnue, savePath);
    }
  }

  print('\n✅ Training complete.');
}

// -----------------------------
// Overfit Single Position
// -----------------------------
Future<void> runOverfitSinglePosition({
  required ChessWithNNUE game,
  required String fen,
  required double targetCp, // centipawns
  int epochs = 200,
  int batchSize = 32,
  double lr = 0.001,
  double earlyStopAbsErrorCp = 1.0,
  String savePath = 'chess_model_overfit.json',
}) async {
  final batch = _buildOverfitBatch(game, fen, targetCp, batchSize);

  final baseline = _predictCp(game, fen);
  final baselineMse = (baseline - targetCp) * (baseline - targetCp);
  print(
    "Baseline -> pred: ${baseline.toStringAsFixed(2)} cp | "
    "target: ${targetCp.toStringAsFixed(2)} cp | mse: ${baselineMse.toStringAsFixed(2)}",
  );

  double pred = baseline;
  for (int e = 1; e <= epochs; e++) {
    game.nnue.trainBatch(batch, lr);

    pred = _predictCp(game, fen);
    final err = pred - targetCp;
    final mse = err * err;

    if (e <= 20 || e % 5 == 0) {
      print(
        "Epoch $e -> pred: ${pred.toStringAsFixed(2)} cp | "
        "err: ${err.toStringAsFixed(2)} cp | mse: ${mse.toStringAsFixed(2)}",
      );
    }

    if (err.abs() <= earlyStopAbsErrorCp) {
      print("Early stop at epoch $e: abs error ≤ ${earlyStopAbsErrorCp} cp");
      break;
    }
  }

  final finalPred = _predictCp(game, fen);
  final finalErr = finalPred - targetCp;
  final finalMse = finalErr * finalErr;
  print(
    "Final   -> pred: ${finalPred.toStringAsFixed(2)} cp | "
    "err: ${finalErr.toStringAsFixed(2)} cp | mse: ${finalMse.toStringAsFixed(2)}",
  );

  await _trySave(game.nnue, savePath);
}

// -----------------------------
// Entry point with modes
// -----------------------------
Future<void> main(List<String> args) async {
  final A = _Args.parse(args);

  final mode = A.getString('mode', 'train'); // 'train' or 'overfit'
  final loadPath = A.getString('loadPath', ''); // optional load

  // Create engine and helpers
  final game = ChessWithNNUE();
  final trainer = NNUETrainer(game, batchSize: A.getInt('batch', 32));
  final searcher = NNUESearcher(game);

  // Optional: load existing model
  if (loadPath.isNotEmpty) {
    await _tryLoad(game.nnue, loadPath);
  }

  if (mode == 'overfit') {
    // Overfit mode params
    final fen = A.getString(
      'fen',
      'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3',
    );

    final useSearchLabel = A.getBool('useSearchLabel', false);
    double targetCp = A.getDouble('target', 0.27); // used if !useSearchLabel
    final labelDepth = A.getInt('labelDepth', 3);

    if (useSearchLabel) {
      final res = searcher.search(labelDepth);
      targetCp = res.score;
      print(
        "Label (search depth $labelDepth): ${targetCp.toStringAsFixed(2)} cp",
      );
    } else {
      print("Label (fixed): ${targetCp.toStringAsFixed(2)} cp");
    }

    final epochs = A.getInt('epochs', 200);
    final batchSize = A.getInt('batch', 32);
    final lr = A.getDouble('lr', 0.001);
    final tol = A.getDouble('tol', 1.0); // abs error cp
    final savePath = A.getString('savePath', 'chess_model_overfit.json');

    await runOverfitSinglePosition(
      game: game,
      fen: fen,
      targetCp: targetCp,
      epochs: epochs,
      batchSize: batchSize,
      lr: lr,
      earlyStopAbsErrorCp: tol,
      savePath: savePath,
    );
    return;
  }

  // Default: general training mode
  final iterations = A.getInt('iters', 50);
  final samplesPerIter = A.getInt('samples', 64);
  final labelDepth = A.getInt('labelDepth', 2);
  final evalDepth = A.getInt('evalDepth', 3);
  final baseLr = A.getDouble('baseLr', 0.001);
  final schedule = _parseSchedule(A.getString('schedule', 'fixed'));
  final stepEvery = A.getInt('stepEvery', 25);
  final gamma = A.getDouble('gamma', 0.5);
  final autosave = A.getBool('autosave', true);
  final savePath = A.getString('savePath', 'chess_model_iter.json');
  final saveEvery = A.getInt('saveEvery', 10);

  await runTrainIterations(
    game: game,
    trainer: trainer,
    searcher: searcher,
    iterations: iterations,
    samplesPerIter: samplesPerIter,
    labelDepth: labelDepth,
    evalDepth: evalDepth,
    baseLr: baseLr,
    schedule: schedule,
    stepEvery: stepEvery,
    gamma: gamma,
    autosave: autosave,
    savePath: savePath,
    saveEvery: saveEvery,
  );
}
