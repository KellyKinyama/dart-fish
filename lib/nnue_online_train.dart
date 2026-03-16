// lib/nnue_online_train.dart
import 'dart:math';
import 'chess3.dart';
import 'chess_with_nnue.dart'; // your wrapper that uses NNUERef
import 'nnue_reference.dart'; // NNUERef.cloneFrom
import 'nnue_logic_batch2.dart'; // TrainingPosition
import 'nnue_persistence_ref.dart'; // serializer for NNUERef (JSON)

/// Deep-copy board so samples are immutable (undo/make won't mutate stored samples).
List<Piece?> _cloneBoard(List<Piece?> board) {
  final out = List<Piece?>.filled(board.length, null);
  for (int i = 0; i < board.length; i++) {
    final p = board[i];
    if (p != null) out[i] = Piece(p.type, p.color);
  }
  return out;
}

/// Evaluate with a **frozen teacher** (fresh accumulator per call).
double _evalWithTeacher(NNUERef teacher, List<Piece?> board, Color stm) {
  final acc = teacher.newAccumulator();
  teacher.refreshAccumulator(acc, board);
  return teacher.evaluate(acc, stm); // cp
}

Future<void> main() async {
  final game = ChessWithNNUE();
  final rng = Random(42);

  const modelPath = 'chess_model_online.json';

  // Try to load prior weights
  final ok = await NNUESerializer.load(game.nnue, modelPath);
  print(
    ok ? "Loaded NNUE model from $modelPath" : "Starting from random weights.",
  );

  // --- Online training settings ---
  const int plies = 80; // how many half-moves to play/train
  const double lr = 5e-4; // small learning rate for stability
  const int teacherLag = 16; // refresh teacher every N moves
  const bool verbose = true;

  // Teacher snapshot (lagged)
  var teacher = NNUERef.cloneFrom(game.nnue);
  var movesSinceTeacher = 0;

  // A running validation set (label once with teacher, keep fixed)
  final validation = <TrainingPosition>[];
  {
    // Collect 32 positions for validation
    for (int i = 0; i < 32; i++) {
      final boardCopy = _cloneBoard(game.board);
      final stm = game.turn;
      final targetCp = _evalWithTeacher(teacher, boardCopy, stm);
      validation.add(TrainingPosition(boardCopy, stm, targetCp));

      final moves = game.generate_moves();
      if (moves.isNotEmpty) {
        final mv = moves[rng.nextInt(moves.length)];
        game.make_move(mv);
      } else {
        game.reset();
      }
    }
    // reset to start
    game.reset();
    game.fullNnueRefresh();
  }

  // Helper: compute validation MSE in network units (cp/400)
  double _valMseNet() {
    double sse = 0.0;
    for (final tp in validation) {
      final acc = game.nnue.newAccumulator();
      game.nnue.refreshAccumulator(acc, tp.board);
      final predCp = game.nnue.evaluate(acc, tp.turn);
      final pred = predCp / NNUERef.SCALE_CP;
      final target = tp.target / NNUERef.SCALE_CP;
      sse += (pred - target) * (pred - target);
    }
    return sse / (validation.isEmpty ? 1 : validation.length);
  }

  print(
    "--- Online NNUE training (post-move update with lagged teacher=$teacherLag) ---",
  );
  print("Initial eval (CP): ${game.nnueEvaluation.toStringAsFixed(4)}");

  for (int ply = 1; ply <= plies; ply++) {
    // 1) Pick a random legal move (self-play/random playout)
    final moves = game.generate_moves();
    if (moves.isEmpty) {
      // Game ended; reset to start
      game.reset();
      game.fullNnueRefresh();
      continue;
    }
    final mv = moves[rng.nextInt(moves.length)];

    // 2) Apply move (this updates accumulator incrementally in your wrapper)
    game.make_move(mv);

    // 3) BEFORE: student prediction on current position (CP)
    final predBefore = game.nnueEvaluation; // cp

    // 4) TARGET: label with the frozen teacher (CP)
    final boardCopy = _cloneBoard(game.board);
    final stm = game.turn;
    final targetCp = _evalWithTeacher(teacher, boardCopy, stm);

    // 5) TRAIN: one-step online update on the single sample
    final sample = TrainingPosition(boardCopy, stm, targetCp);
    game.nnue.trainBatch([sample], lr);

    // 6) IMPORTANT: refresh accumulator after weights change
    game.fullNnueRefresh();

    // 7) AFTER: new prediction (CP)
    final predAfter = game.nnueEvaluation;

    // 8) Log
    if (verbose) {
      final errBeforeCp = predBefore - targetCp;
      final errAfterCp = predAfter - targetCp;
      final beforeNet = errBeforeCp / NNUERef.SCALE_CP;
      final afterNet = errAfterCp / NNUERef.SCALE_CP;
      print(
        "ply $ply: predBefore=${predBefore.toStringAsFixed(3)} cp | "
        "target=${targetCp.toStringAsFixed(3)} cp | "
        "predAfter=${predAfter.toStringAsFixed(3)} cp | "
        "mseBeforeNet=${(beforeNet * beforeNet).toStringAsFixed(6)} "
        "mseAfterNet=${(afterNet * afterNet).toStringAsFixed(6)}",
      );
    }

    // 9) Periodically refresh the teacher (lag)
    movesSinceTeacher++;
    if (movesSinceTeacher >= teacherLag) {
      teacher = NNUERef.cloneFrom(game.nnue);
      movesSinceTeacher = 0;
      final valMse = _valMseNet();
      print(
        "  [teacher refresh] validation MSE (net): ${valMse.toStringAsFixed(6)}",
      );
    }
  }

  print("Final eval (CP): ${game.nnueEvaluation.toStringAsFixed(4)}");

  // Save the model
  try {
    await NNUESerializer.save(game.nnue, modelPath);
    print("Saved NNUE model to $modelPath");
  } catch (e) {
    print("Save failed: $e");
  }
}
