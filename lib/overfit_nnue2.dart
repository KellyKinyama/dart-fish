// lib/nnue_train_frozen_teacher.dart
import 'dart:math' show Random, min;
import 'package:dart_fish/chess_nnue3.dart'; // ChessWithNNUE (student)
import 'package:dart_fish/alphabeta_train2.dart'; // NNUESearcher, NNUETrainer (we'll extend)
import 'package:dart_fish/nnue_logic_batch2.dart'; // NNUE, TrainingPosition, NnueAccumulator
import 'package:dart_fish/nnue_persistence2.dart';

import 'chess3.dart'; // NNUESerializer

// -----------------------------
// Utilities
// -----------------------------

/// Deep copy Piece list so TrainingPosition is immutable.
List<Piece?> _cloneBoard(List<Piece?> board) {
  final out = List<Piece?>.filled(board.length, null);
  for (int i = 0; i < board.length; i++) {
    final p = board[i];
    if (p != null) out[i] = Piece(p.type, p.color);
  }
  return out;
}

/// Clone NNUE *weights & biases only* (moments reset to zero in dst).
NNUE cloneNNUE(NNUE src) {
  final dst = NNUE(); // random init, will be overwritten
  // Copy l0
  for (int i = 0; i < NNUE.NUM_FEATURES; i++) {
    final sw = src.l0.weights[i];
    final dw = dst.l0.weights[i];
    for (int j = 0; j < NNUE.M; j++) {
      dw[j] = sw[j];
    }
  }
  for (int j = 0; j < NNUE.M; j++) {
    dst.l0.bias[j] = src.l0.bias[j];
  }

  // Copy l1
  for (int i = 0; i < 2 * NNUE.M; i++) {
    final sw = src.l1.weights[i];
    final dw = dst.l1.weights[i];
    for (int j = 0; j < NNUE.K; j++) {
      dw[j] = sw[j];
    }
  }
  for (int j = 0; j < NNUE.K; j++) {
    dst.l1.bias[j] = src.l1.bias[j];
  }

  // Copy l2
  for (int i = 0; i < NNUE.K; i++) {
    dst.l2.weights[i][0] = src.l2.weights[i][0];
  }
  dst.l2.bias[0] = src.l2.bias[0];

  // Moments remain zero in dst (fresh optimizer state)
  return dst;
}

/// Build an accumulator for a given NNUE and a 0x88 board (64-features mapping).
NnueAccumulator buildAccumulatorFor(NNUE net, List<Piece?> board) {
  final acc = NnueAccumulator(NNUE.M);

  // Find king squares in 0x88
  int wK0x88 = board.indexWhere(
    (p) => p?.type == PieceType.KING && p?.color == Color.WHITE,
  );
  int bK0x88 = board.indexWhere(
    (p) => p?.type == PieceType.KING && p?.color == Color.BLACK,
  );
  if (wK0x88 == -1 || bK0x88 == -1) {
    // Bias only
    acc.refresh(net.l0, const <int>[], 0);
    acc.refresh(net.l0, const <int>[], 1);
    return acc;
  }
  int to64(int s) => ((s & 7) | ((s >> 4) << 3));

  final wK = to64(wK0x88);
  final bK = to64(bK0x88);

  final wf = <int>[];
  final bf = <int>[];

  for (int sq = 0; sq < 128; ++sq) {
    if ((sq & 0x88) != 0) continue;
    final p = board[sq];
    if (p == null || p.type == PieceType.KING) continue;
    final s64 = to64(sq);
    final wi = net.getHalfKPIndex(wK, s64, p, false);
    final bi = net.getHalfKPIndex(bK, s64, p, true);
    if (wi != -1) wf.add(wi);
    if (bi != -1) bf.add(bi);
  }
  acc.refresh(net.l0, wf, 0);
  acc.refresh(net.l0, bf, 1);
  return acc;
}

/// Evaluate a position in CP using a *frozen NNUE* (teacher) directly from a 0x88 board + stm.
double evalWithFrozen(NNUE teacher, List<Piece?> board, Color stm) {
  final acc = buildAccumulatorFor(teacher, board);
  return teacher.evaluate(acc, stm);
}

// -----------------------------
// Frozen-Teacher Trainer
// -----------------------------

class FrozenTeacherTrainer {
  final ChessWithNNUE student; // where we generate games & train
  final Random _rng;
  final int batchSize;

  FrozenTeacherTrainer(this.student, {this.batchSize = 32, Random? rng})
    : _rng = rng ?? Random();

  /// Generate a batch labeled by a *frozen* teacher (not the student's live NNUE).
  List<TrainingPosition> generateLabeledBatch({
    required NNUE frozenTeacher,
    required int numPositions,
    int randomMovePlayout = 1, // randomize a little between samples
  }) {
    final batch = <TrainingPosition>[];

    for (int i = 0; i < numPositions; i++) {
      // Snapshot current board
      final boardCopy = _cloneBoard(student.board);
      final stm = student.turn;

      // Label with frozen teacher
      final targetCp = evalWithFrozen(frozenTeacher, boardCopy, stm);

      // Add to batch
      batch.add(TrainingPosition(boardCopy, stm, targetCp));

      // Playout: make 1 random legal move to diversify next sample
      final moves = student.generate_moves();
      if (moves.isNotEmpty) {
        final m = moves[_rng.nextInt(moves.length)];
        student.make_move(m);
      } else {
        student.reset();
      }
    }
    return batch;
  }

  /// Compute MSE (in cp^2) for a batch using the student's current NNUE
  double mseOnBatch(List<TrainingPosition> batch) {
    double sse = 0.0;
    for (final tp in batch) {
      final pred = evalWithFrozen(student.nnue, tp.board, tp.turn);
      final err = pred - tp.target;
      sse += err * err;
    }
    return sse / batch.length;
  }

  /// Compute MSE on a small held-out validation set (positions+labels fixed)
  double mseOnValidation(List<TrainingPosition> val) {
    return mseOnBatch(val);
  }
}

// -----------------------------
// Main training loop (frozen teacher)
// -----------------------------

Future<void> main() async {
  final student = ChessWithNNUE();
  final trainer = FrozenTeacherTrainer(student, batchSize: 32);
  final searcher = NNUESearcher(student); // optional, for periodic eval at root

  const modelPath = 'chess_model_iter_frozen.json';

  // Optional: load prior weights
  await _tryLoad(student.nnue, modelPath);

  // --- Build a tiny held-out validation set (frozen teacher labels) ---
  // Use the initial snapshot as teacher for val (kept fixed throughout).
  final teacherForVal = cloneNNUE(student.nnue);
  final validation = trainer.generateLabeledBatch(
    frozenTeacher: teacherForVal,
    numPositions: 64,
    randomMovePlayout: 2,
  );

  // --- Hyperparams ---
  const iterations = 50;
  const samplesPerIter = 64;
  const baseLr = 0.001;

  print('--- NNUE Training w/ Frozen Teacher ---');
  print('iters=$iterations, samples=$samplesPerIter, lr=$baseLr');

  for (int it = 1; it <= iterations; it++) {
    print('\n--- Iteration $it/$iterations ---');

    // 1) Freeze teacher snapshot
    final frozenTeacher = cloneNNUE(student.nnue);

    // 2) Generate labeled batch with the frozen teacher
    final batch = trainer.generateLabeledBatch(
      frozenTeacher: frozenTeacher,
      numPositions: samplesPerIter,
      randomMovePlayout: 1,
    );

    // 3) Log pre-train MSE (optional)
    final preMse = trainer.mseOnBatch(batch);
    print('Pre-train MSE (batch): ${preMse.toStringAsFixed(2)} cp^2');

    // 4) Train student on this batch
    //    We’ll feed the batch directly to student.nnue.trainBatch
    student.nnue.trainBatch(batch, baseLr);

    // 5) Post-train MSE
    final postMse = trainer.mseOnBatch(batch);
    print('Post-train MSE (batch): ${postMse.toStringAsFixed(2)} cp^2');

    // 6) Validation MSE (fixed held-out set)
    final valMse = trainer.mseOnValidation(validation);
    print('Validation MSE: ${valMse.toStringAsFixed(2)} cp^2');

    // 7) Optional: check root eval trend (not a loss; just a reference)
    student.reset();
    final evalRes = searcher.search(3);
    print(
      'Root eval @depth3: ${evalRes.score.toStringAsFixed(2)} cp | nodes: ${evalRes.nodes}',
    );

    // 8) Save every 10 iters
    if (it % 10 == 0 || it == iterations) {
      await _trySave(student.nnue, modelPath);
    }
  }

  print('\n✅ Training complete.');
}

// -----------------------------
// Persistence helpers
// -----------------------------
Future<void> _tryLoad(NNUE nnue, String path) async {
  try {
    await NNUESerializer.load(nnue, path);
    print("✅ Loaded NNUE model from $path");
  } catch (e) {
    print(
      "⚠️ No weights file found or load failed at $path. Starting fresh. ($e)",
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
