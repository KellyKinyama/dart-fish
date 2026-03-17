// lib/nnue_static_dataset.dart
import 'dart:math';
import 'chess3.dart';
import 'chess_with_nnue.dart';
import 'nnue_logic_batch2.dart';
import 'nnue_reference.dart';
import 'static_labeler.dart';

class StaticDatasetBuilder {
  final ARSPositionSampler sampler = ARSPositionSampler();

  List<TrainingPosition> generate(
    ChessWithNNUE game, {
    int numPositions = 256,
    int playoutDepth = 6,
  }) {
    final samples = <TrainingPosition>[];

    print("\n==============================");
    print("  DATASET GENERATION STARTED");
    print("  Target samples: $numPositions");
    print("  Playout depth: $playoutDepth");
    print(
      "  Using ARS sampler: ${sampler.numDirections} dirs, σ=${sampler.sigma}",
    );
    print("==============================\n");

    int attempt = 0;

    while (samples.length < numPositions) {
      attempt++;

      print(
        ">> SAMPLE ${samples.length + 1}/$numPositions  (attempt $attempt)",
      );

      final t0 = DateTime.now().millisecondsSinceEpoch;

      game.reset();
      print("   - Board reset. Starting ARS playout...");

      sampler.playoutWithARS(game, playoutDepth);

      // Skip terminal positions
      if (game.generate_moves().isEmpty) {
        print("   - Position is terminal → SKIPPED\n");
        continue;
      }

      final fen = game.fen;
      final targetCp = staticEvalCpFromFen(fen);
      final dt = DateTime.now().millisecondsSinceEpoch - t0;

      print("   - ARS playout completed in ${dt}ms");
      print("   - Resulting FEN: $fen");
      print("   - Static CP target: $targetCp");
      print("   - Sample ACCEPTED.\n");

      samples.add(
        TrainingPosition(List<Piece?>.from(game.board), game.turn, targetCp),
      );
    }

    print("==============================");
    print("  DATASET GENERATION COMPLETE");
    print("  Produced $numPositions samples");
    print("  Returning to trainer loop...");
    print("==============================\n");

    return samples;
  }
}

class ARSPositionSampler {
  final Random rng = Random();
  final int numDirections;
  final double sigma;

  ARSPositionSampler({this.numDirections = 8, this.sigma = 0.002});

  void playoutWithARS(ChessWithNNUE game, int plies) {
    print("Starting ARS playout for $plies plies...");

    for (int t = 0; t < plies; t++) {
      print("\n--- PLY $t ---");

      final moves = game.generate_moves();
      if (moves.isEmpty) {
        print("No legal moves (terminal).");
        return;
      }

      print(
        "Legal moves (${moves.length}): ${moves.map((m) => m.fromAlgebraic + m.toAlgebraic).join(', ')}",
      );

      Move? bestMove;
      double bestScore = -1e18;

      for (int d = 0; d < numDirections; d++) {
        print("  Direction $d:");

        final clone = NNUERef.cloneFrom(game.nnue);
        _applyPerturbation(clone, sigma);
        print("    Applied perturbation (σ = $sigma).");

        for (final mv in moves) {
          final lan = mv.fromAlgebraic + mv.toAlgebraic;
          game.make_move(mv);

          final acc = clone.newAccumulator();
          clone.refreshAccumulator(acc, game.board);

          final score = clone.evaluate(acc, game.turn);

          print("    Move $lan -> score: ${score.toStringAsFixed(3)}");

          game.undo_move();

          if (score > bestScore) {
            bestScore = score;
            bestMove = mv;
            print("    -> New best move: $lan (score $bestScore)");
          }
        }
      }

      if (bestMove == null) {
        bestMove = moves[rng.nextInt(moves.length)];
        print(
          "  No ARS best found. Choosing RAND: ${bestMove.fromAlgebraic}${bestMove.toAlgebraic}",
        );
      } else {
        print(
          "  BEST move selected: ${bestMove.fromAlgebraic}${bestMove.toAlgebraic}   (score $bestScore)",
        );
      }

      game.make_move(bestMove);
    }

    print("ARS playout completed.\n");
  }

  void _applyPerturbation(NNUERef model, double s) {
    final rng = Random();

    for (final layer in [model.l1, model.l2]) {
      for (int c = 0; c < layer.inSize; c++) {
        for (int r = 0; r < layer.outSize; r++) {
          layer.weights[c][r] += (rng.nextDouble() * 2 - 1) * s;
        }
      }
      for (int r = 0; r < layer.outSize; r++) {
        layer.bias[r] += (rng.nextDouble() * 2 - 1) * s;
      }
    }
  }
}
