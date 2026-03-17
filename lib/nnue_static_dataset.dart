// lib/nnue_static_dataset.dart
import 'dart:math';
import 'chess3.dart';
import 'chess_with_nnue.dart'; // Your wrapper that extends chess3.Chess
// import 'engine.dart';
import 'nnue_logic_batch2.dart'; // TrainingPosition
import 'static_labeler.dart'; // staticEvalCpFromFen

class StaticDatasetBuilder {
  final Random rng = Random();

  List<TrainingPosition> generate(
    ChessWithNNUE game, {
    int numPositions = 16,
    int playoutDepth = 4,
  }) {
    final samples = <TrainingPosition>[];

    while (samples.length < numPositions) {
      // Reset to clean starting position
      game.reset();

      bool terminal = false;

      // Random playout
      for (int j = 0; j < playoutDepth; j++) {
        final moves = game.generate_moves();

        if (moves.isEmpty) {
          // Checkmate or stalemate → skip this position
          terminal = true;
          break;
        }

        game.make_move(moves[rng.nextInt(moves.length)]);
      }

      if (terminal) {
        // Try again, do NOT add this sample
        continue;
      }

      // At this point the position is valid → label it
      final fen = game.fen;
      final targetCp = staticEvalCpFromFen(fen);

      final boardCopy = List<Piece?>.from(game.board);
      final stm = game.turn;

      samples.add(TrainingPosition(boardCopy, stm, targetCp));
    }

    return samples;
  }
}
