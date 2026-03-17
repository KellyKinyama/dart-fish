// lib/nnue_static_dataset.dart
import 'dart:math';
import 'dart:typed_data';
import 'chess3.dart';
import 'chess_with_nnue.dart';
import 'nnue_logic_batch2.dart';
import 'nnue_reference.dart';
import 'static_labeler.dart';

// ======================================================
// STATIC DATASET BUILDER (unchanged structure)
// ======================================================
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

// ======================================================
// ADVERSARIAL ARS LEVEL 3 SAMPLER
// ======================================================
//
// Upgrades from Level 2:
//   ✔ White maximizes score
//   ✔ Black minimizes score
//   ✔ NNUE evaluation flipped for black
//   ✔ Still uses ± perturbations
//   ✔ Uses top‑b direction ranking
//   ✔ Temporary boards only (correct)
//
// This produces realistic adversarial self-play positions.
//
// ======================================================

class ARSPositionSampler {
  final Random rng = Random();

  final int numDirections;
  final int topB;
  final double sigma;

  ARSPositionSampler({this.numDirections = 8, int? topB, this.sigma = 0.002})
    : topB = topB ?? (8 ~/ 2);

  void playoutWithARS(ChessWithNNUE game, int plies) {
    print("Adversarial ARS Level 3 playout: dirs=$numDirections  topB=$topB");

    for (int ply = 0; ply < plies; ply++) {
      print("\n--- PLY $ply ---");

      final moves = game.generate_moves();
      if (moves.isEmpty) {
        print("Terminal position during playout.");
        return;
      }

      print(
        "Legal moves: ${moves.map((m) => m.fromAlgebraic + m.toAlgebraic).join(', ')}",
      );

      final List<_DirectionEval> dirs = [];

      // -------------------------------------------------
      // 1) SAMPLE ± directions
      // -------------------------------------------------
      for (int d = 0; d < numDirections; d++) {
        print("  Direction $d...");

        final perturb = _RandomPerturb(game.nnue, sigma);

        final plus = NNUERef.cloneFrom(game.nnue);
        perturb.applyTo(plus, sign: 1);

        final minus = NNUERef.cloneFrom(game.nnue);
        perturb.applyTo(minus, sign: -1);

        double bestPlus = -1e18;
        double bestMinus = -1e18;

        for (final mv in moves) {
          // +delta / temp board
          {
            final temp = ChessWithNNUE();
            temp.load(game.fen);
            temp.make_move(mv);

            final acc = plus.newAccumulator();
            plus.refreshAccumulator(acc, temp.board);

            double score = plus.evaluate(acc, temp.turn);
            if (temp.turn == Color.BLACK) score = -score;

            if (score > bestPlus) bestPlus = score;
          }

          // -delta / temp board
          {
            final temp = ChessWithNNUE();
            temp.load(game.fen);
            temp.make_move(mv);

            final acc = minus.newAccumulator();
            minus.refreshAccumulator(acc, temp.board);

            double score = minus.evaluate(acc, temp.turn);
            if (temp.turn == Color.BLACK) score = -score;

            if (score > bestMinus) bestMinus = score;
          }
        }

        print("    +delta best score = $bestPlus");
        print("    -delta best score = $bestMinus");

        dirs.add(_DirectionEval(perturb, bestPlus, bestMinus));
      }

      // rank by max of plus/minus
      dirs.sort((a, b) => max(b.plus, b.minus).compareTo(max(a.plus, a.minus)));

      final bestDirs = dirs.take(topB).toList();
      print("  Top-$topB directions selected.");

      // -------------------------------------------------
      // 2) FINAL ADVERSARIAL MOVE SCORING
      // -------------------------------------------------

      Move? bestMove;
      double bestMoveScore = (game.turn == Color.WHITE) ? -1e18 : 1e18;

      for (final mv in moves) {
        double scoreSum = 0;

        for (final dir in bestDirs) {
          final sign = (dir.plus >= dir.minus) ? 1 : -1;

          final clone = NNUERef.cloneFrom(game.nnue);
          dir.perturb.applyTo(clone, sign: sign);

          final temp = ChessWithNNUE();
          temp.load(game.fen);
          temp.make_move(mv);

          final acc = clone.newAccumulator();
          clone.refreshAccumulator(acc, temp.board);

          double score = clone.evaluate(acc, temp.turn);
          if (temp.turn == Color.BLACK) score = -score;

          scoreSum += score;
        }

        final meanScore = scoreSum / bestDirs.length;
        print("    Move ${mv.fromAlgebraic}${mv.toAlgebraic} → $meanScore");

        if (game.turn == Color.WHITE) {
          // maximize
          if (meanScore > bestMoveScore) {
            bestMoveScore = meanScore;
            bestMove = mv;
          }
        } else {
          // minimize
          if (meanScore < bestMoveScore) {
            bestMoveScore = meanScore;
            bestMove = mv;
          }
        }
      }

      bestMove ??= moves[rng.nextInt(moves.length)];

      print(
        "  BEST MOVE = ${bestMove.fromAlgebraic}${bestMove.toAlgebraic}  (score $bestMoveScore)\n",
      );

      game.make_move(bestMove);
    }
  }
}

// ======================================================
// SUPPORT CLASSES
// ======================================================

class _DirectionEval {
  final _RandomPerturb perturb;
  final double plus;
  final double minus;
  _DirectionEval(this.perturb, this.plus, this.minus);
}

class _RandomPerturb {
  final Float64List l1W;
  final Float64List l1B;
  final Float64List l2W;
  final Float64List l2B;

  _RandomPerturb(NNUERef base, double sigma)
    : l1W = Float64List(base.l1.inSize * base.l1.outSize),
      l1B = Float64List(base.l1.outSize),
      l2W = Float64List(base.l2.inSize * base.l2.outSize),
      l2B = Float64List(base.l2.outSize) {
    final r = Random();
    for (int i = 0; i < l1W.length; i++)
      l1W[i] = (r.nextDouble() * 2 - 1) * sigma;
    for (int i = 0; i < l1B.length; i++)
      l1B[i] = (r.nextDouble() * 2 - 1) * sigma;
    for (int i = 0; i < l2W.length; i++)
      l2W[i] = (r.nextDouble() * 2 - 1) * sigma;
    for (int i = 0; i < l2B.length; i++)
      l2B[i] = (r.nextDouble() * 2 - 1) * sigma;
  }

  void applyTo(NNUERef model, {required int sign}) {
    final s = sign.toDouble();

    int k = 0;
    for (int c = 0; c < model.l1.inSize; c++) {
      for (int r = 0; r < model.l1.outSize; r++) {
        model.l1.weights[c][r] += s * l1W[k++];
      }
    }
    for (int r = 0; r < model.l1.outSize; r++) {
      model.l1.bias[r] += s * l1B[r];
    }

    k = 0;
    for (int c = 0; c < model.l2.inSize; c++) {
      for (int r = 0; r < model.l2.outSize; r++) {
        model.l2.weights[c][r] += s * l2W[k++];
      }
    }
    for (int r = 0; r < model.l2.outSize; r++) {
      model.l2.bias[r] += s * l2B[r];
    }
  }
}
