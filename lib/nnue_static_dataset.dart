// lib/nnue_static_dataset.dart
import 'dart:math';
import 'chess3.dart';
import 'chess_with_nnue.dart'; // Your wrapper that extends chess3.Chess
import 'nnue_logic_batch2.dart'; // TrainingPosition
import 'static_labeler.dart'; // staticEvalCpFromFen

class StaticDatasetBuilder {
  final Random rng = Random();

  /// Generate labeled samples using random playouts from the current position.
  /// Labels come from the static evaluator via FEN (independent of NNUE).
  ///
  /// - numPositions: how many samples to create
  /// - playoutDepth: how many random moves to play before labeling
  ///
  /// Returns a list of TrainingPosition(boardCopy, sideToMove, targetCp).
  List<TrainingPosition> generate(
    ChessWithNNUE game, {
    int numPositions = 256,
    int playoutDepth = 4,
  }) {
    final samples = <TrainingPosition>[];

    for (int i = 0; i < numPositions; i++) {
      // Start from the initial position (or current position if you prefer)
      game.reset();

      // Random playout to diversify
      for (int j = 0; j < playoutDepth; j++) {
        final moves = game.generate_moves();
        if (moves.isEmpty) break;
        final mv = moves[rng.nextInt(moves.length)];
        game.make_move(mv);
      }

      // Label from static evaluator using FEN
      final fen = game.fen; // <- string, no cross-type issues
      final targetCp = staticEvalCpFromFen(fen);

      // Snapshot board & side-to-move for the sample
      final boardCopy = List<Piece?>.from(game.board);
      final stm = game.turn;

      samples.add(TrainingPosition(boardCopy, stm, targetCp));
    }

    return samples;
  }
}
