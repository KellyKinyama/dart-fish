// lib/nnue_frozen_train.dart
import 'dart:math';
import 'chess3.dart';
import 'chess_with_nnue.dart';
import 'nnue_reference.dart';
import 'nnue_logic_batch2.dart';
import 'nnue_persistence_ref.dart';

/// Deep-copy board so samples are immutable across make/undo.
List<Piece?> cloneBoard(List<Piece?> board) {
  final out = List<Piece?>.filled(board.length, null);
  for (int i = 0; i < board.length; i++) {
    final p = board[i];
    if (p != null) out[i] = Piece(p.type, p.color);
  }
  return out;
}

/// Evaluate in CP with a frozen NNUERef (teacher) using a fresh accumulator.
double evalWithTeacher(NNUERef teacher, List<Piece?> board, Color stm) {
  final acc = teacher.newAccumulator();
  teacher.refreshAccumulator(acc, board);
  return teacher.evaluate(acc, stm);
}

class FrozenTeacherTrainer {
  final ChessWithNNUE student;
  final Random _rng;
  final int randomPlayout; // number of random moves to diversify positions

  FrozenTeacherTrainer(this.student, {Random? rng, this.randomPlayout = 1})
    : _rng = rng ?? Random();

  /// Generate a batch labeled by a *frozen teacher* (no search, direct eval).
  List<TrainingPosition> genBatchWithTeacher({
    required NNUERef teacher,
    required int numPositions,
  }) {
    final batch = <TrainingPosition>[];
    for (int i = 0; i < numPositions; i++) {
      final boardCopy = cloneBoard(student.board);
      final stm = student.turn;
      final targetCp = evalWithTeacher(teacher, boardCopy, stm);
      batch.add(TrainingPosition(boardCopy, stm, targetCp));

      // playout some random moves to diversify next sample
      var moves = student.generate_moves();
      if (moves.isEmpty) {
        student.reset();
        continue;
      }
      for (int s = 0; s < randomPlayout; s++) {
        moves = student.generate_moves();
        if (moves.isEmpty) break;
        final mv = moves[_rng.nextInt(moves.length)];
        student.make_move(mv);
      }
    }
    return batch;
  }

  /// MSE in *network units* (net = cp / 400).
  double mseNet(NNUERef model, List<TrainingPosition> samples) {
    if (samples.isEmpty) return 0.0;
    double sse = 0.0;
    for (final tp in samples) {
      final predCp = model.evalBoardCp(tp.board, tp.turn);
      final pred = predCp / NNUERef.SCALE_CP;
      final target = tp.target / NNUERef.SCALE_CP;
      final err = pred - target;
      sse += err * err;
    }
    return sse / samples.length;
  }

  /// MSE in *centipawns squared* for reporting.
  double mseCp(NNUERef model, List<TrainingPosition> samples) {
    if (samples.isEmpty) return 0.0;
    double sse = 0.0;
    for (final tp in samples) {
      final predCp = model.evalBoardCp(tp.board, tp.turn);
      final err = predCp - tp.target;
      sse += err * err;
    }
    return sse / samples.length;
  }
}

Future<void> main() async {
  final student = ChessWithNNUE();
  final trainer = FrozenTeacherTrainer(student, randomPlayout: 1);
  final rng = Random(123);

  const modelPath = 'chess_model_frozen.json';
  final ok = await NNUESerializer.load(student.nnue, modelPath);
  print(ok ? "Loaded model from $modelPath" : "Starting from random weights.");

  // Build a validation set (teacher-labeled once and kept fixed)
  print("Building validation set...");
  final teacherForVal = NNUERef.cloneFrom(student.nnue);
  final validation = <TrainingPosition>[];
  {
    // 64 positions, 1-ply random diversification
    const valCount = 64;
    for (int i = 0; i < valCount; i++) {
      final boardCopy = cloneBoard(student.board);
      final stm = student.turn;
      final targetCp = evalWithTeacher(teacherForVal, boardCopy, stm);
      validation.add(TrainingPosition(boardCopy, stm, targetCp));

      final moves = student.generate_moves();
      if (moves.isNotEmpty) {
        student.make_move(moves[rng.nextInt(moves.length)]);
      } else {
        student.reset();
      }
    }
    // return to start for training loop
    student.reset();
  }

  // Hyperparameters
  const iters = 10;
  const batchSize = 64;
  const lr = 0.0005; // start smaller for stability with frozen teacher

  print("--- Frozen-Teacher NNUE Training ---");
  for (int it = 1; it <= iters; it++) {
    print("\n--- Iteration $it/$iters ---");

    // 1) Snapshot teacher (lag-1)
    final teacher = NNUERef.cloneFrom(student.nnue);

    // 2) Generate teacher-labeled batch (no search, direct teacher eval)
    final batch = trainer.genBatchWithTeacher(
      teacher: teacher,
      numPositions: batchSize,
    );

    // Report pre-train losses
    final preMseNet = trainer.mseNet(student.nnue, batch);
    final preMseCp = trainer.mseCp(student.nnue, batch);
    print(
      "Batch pre-MSE: ${preMseNet.toStringAsFixed(6)} net | ${preMseCp.toStringAsFixed(4)} cp^2",
    );

    // 3) Train student on this teacher-labeled batch
    student.nnue.trainBatch(batch, lr);

    // Report post-train losses
    final postMseNet = trainer.mseNet(student.nnue, batch);
    final postMseCp = trainer.mseCp(student.nnue, batch);
    print(
      "Batch post-MSE: ${postMseNet.toStringAsFixed(6)} net | ${postMseCp.toStringAsFixed(4)} cp^2",
    );

    // 4) Validation against the fixed teacherForVal set
    final valMseNet = trainer.mseNet(student.nnue, validation);
    final valMseCp = trainer.mseCp(student.nnue, validation);
    print(
      "Validation MSE: ${valMseNet.toStringAsFixed(6)} net | ${valMseCp.toStringAsFixed(4)} cp^2",
    );

    // Optional: report start position eval (CP)
    student.reset();
    final acc = student.nnue.newAccumulator();
    student.nnue.refreshAccumulator(acc, student.board);
    final rootCp = student.nnue.evaluate(acc, student.turn);
    print("Root eval (CP): ${rootCp.toStringAsFixed(4)}");

    // 5) Save every iteration (or every N)
    final saved = await NNUESerializer.save(
      student.nnue,
      modelPath,
    ).then((_) => true).catchError((_) => false);
    if (saved) print("Saved -> $modelPath");
  }

  print("\n✅ Frozen-teacher training complete.");
}
