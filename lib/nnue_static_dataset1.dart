// lib/nnue_static_dataset.dart
import 'dart:math';
import 'chess3.dart';
import 'chess_with_nnue.dart';
import 'nnue_logic_batch2.dart';
import 'nnue_reference.dart';
import 'static_labeler.dart';

class StaticDatasetBuilder {
  final Random rng = Random();

  /// Epsilon value for ε-greedy exploration:
  /// ε = chance to play a random move
  /// (1 - ε) = chance to play NNUE-best move
  final double epsilon;

  StaticDatasetBuilder({this.epsilon = 0.85});

  List<TrainingPosition> generate(
    ChessWithNNUE game, {
    int numPositions = 16,
    int playoutDepth = 4,
  }) {
    final samples = <TrainingPosition>[];

    print("=== DATASET GENERATION STARTED ===");
    print("Target samples: $numPositions");
    print("Playout depth: $playoutDepth");
    print("Epsilon: $epsilon");
    print("=================================\n");

    while (samples.length < numPositions) {
      game.reset();
      bool terminal = false;

      print("New episode: starting position.");
      print("=================================");

      for (int ply = 0; ply < playoutDepth; ply++) {
        final moves = game.generate_moves();

        print("\nPLY $ply");
        print("Side to move: ${game.turn}");
        print(
          "Legal moves: ${moves.map((m) => m.fromAlgebraic + m.toAlgebraic).join(', ')}",
        );

        if (moves.isEmpty) {
          print("Terminal detected: no legal moves. Skipping.");
          terminal = true;
          break;
        }

        Move chosen;
        final roll = rng.nextDouble();

        print("Epsilon roll = $roll");

        // ============================================================
        // EPSILON-GREEDY (unchanged)
        // ============================================================
        if (roll < epsilon) {
          // Random exploration move
          chosen = moves[rng.nextInt(moves.length)];

          print(
            "Exploration chosen. Random move = "
            "${chosen.fromAlgebraic}${chosen.toAlgebraic}",
          );
        } else {
          // Exploitation: choose NNUE-best move
          double bestScore = -999999999.0;
          Move? bestMove;

          print("Exploitation chosen. Evaluating all moves...");

          for (final mv in moves) {
            final temp = ChessWithNNUE();
            temp.load(game.fen);

            Color us = temp.turn;

            temp.make_move(mv);

            // NNUE static evaluation for temp board (unchanged)
            final score = -temp.nnue.evaluateBoard(temp.board, us);

            print("Move ${mv.fromAlgebraic}${mv.toAlgebraic} => score $score");

            if (score > bestScore) {
              bestScore = score;
              bestMove = mv;
            }
          }

          print(
            "Best-scoring move: "
            "${bestMove!.fromAlgebraic}${bestMove.toAlgebraic} "
            "score=$bestScore",
          );

          chosen = bestMove!;
        }

        print(
          "Applying move: "
          "${chosen.fromAlgebraic}${chosen.toAlgebraic}\n",
        );

        game.make_move(chosen);
      }

      if (terminal) {
        print("Episode ended in terminal state. Restarting.\n");
        continue;
      }

      // Produce labeled training position (unchanged)
      final fen = game.fen;
      final targetCp = staticEvalCpFromFen(fen);

      print("Final sampled FEN: $fen");
      print("Static label CP: $targetCp");
      print("Sample ACCEPTED.\n");

      samples.add(
        TrainingPosition(List<Piece?>.from(game.board), game.turn, targetCp),
      );
    }

    print("=== DATASET GENERATION COMPLETE ===");
    print("Generated $numPositions samples.\n");

    return samples;
  }
}
