// lib/nnue_static_dataset.dart
import 'dart:math';
import 'dart:typed_data';

import 'chess3.dart';
import 'chess_with_nnue.dart';
import 'nnue_logic_batch2.dart';
import 'nnue_reference.dart';
import 'static_labeler.dart';

// ===============================================================
//   STATIC DATASET BUILDER (moderate debug)
// ===============================================================
class StaticDatasetBuilder {
  final ARSPositionSampler sampler = ARSPositionSampler();

  List<TrainingPosition> generate(
    ChessWithNNUE game, {
    int numPositions = 256,
    int playoutDepth = 6,
  }) {
    final samples = <TrainingPosition>[];

    print("\n===================================================");
    print("  DATASET GENERATION STARTED  (ARS Level 4 — 2-PLY)");
    print("  Target samples: $numPositions");
    print("  Playout depth: $playoutDepth");
    print("===================================================\n");

    int attempt = 0;

    while (samples.length < numPositions) {
      attempt++;
      print(
        "\n>> SAMPLE ${samples.length + 1}/$numPositions (attempt $attempt)",
      );

      final t0 = DateTime.now().millisecondsSinceEpoch;
      game.reset();

      sampler.playoutWithARS(game, playoutDepth);

      if (game.generate_moves().isEmpty) {
        print("  - Terminal → SKIPPED\n");
        continue;
      }

      final fen = game.fen;
      final target = staticEvalCpFromFen(fen);
      final dt = DateTime.now().millisecondsSinceEpoch - t0;

      print("  - Playout done in ${dt}ms");
      print("  - Position FEN: $fen");
      print("  - Static label: $target");
      print("  - Sample added.\n");

      samples.add(
        TrainingPosition(List<Piece?>.from(game.board), game.turn, target),
      );
    }

    print("===================================================");
    print("  DATASET GENERATION COMPLETE — $numPositions samples");
    print("===================================================\n");

    return samples;
  }
}

// ===================================================================
//   ARS LEVEL 4 — ADVERSARIAL + 2-PLY — MODERATE DEBUG
// ===================================================================
class ARSPositionSampler {
  final Random rng = Random();

  final int numDirections;
  final int topB;
  final double sigma;

  ARSPositionSampler({this.numDirections = 8, this.sigma = 0.002, int? topB})
    : topB = topB ?? (8 ~/ 2);

  // -----------------------------
  // Z-score normalization
  // -----------------------------
  List<double> _normalize(List<double> xs) {
    if (xs.isEmpty) return xs;

    double mean = xs.reduce((a, b) => a + b) / xs.length;

    double varSum = 0.0;
    for (final x in xs) {
      varSum += (x - mean) * (x - mean);
    }

    double std = sqrt(varSum / xs.length).abs();
    if (std < 1e-9) std = 1e-9;

    return xs
        .map((x) => ((x - mean) / std).clamp(-3.0, 3.0))
        .toList()
        .cast<double>();
  }

  // ===============================================================
  // MAIN: Robust ARS Level 4 — 2-PLY MINIMAX
  // ===============================================================
  void playoutWithARS(ChessWithNNUE game, int plies) {
    print("  [ARS] Running robust adversarial ARS Level 4 (2-ply)\n");

    for (int ply = 0; ply < plies; ply++) {
      print("---- PLY $ply ----");

      final moves = game.generate_moves();
      if (moves.isEmpty) {
        print("    No legal moves — terminal.\n");
        return;
      }

      final isWhite = (game.turn == Color.WHITE);
      final effSigma = sigma / sqrt(1.0 + ply);
      print("    Effective sigma: $effSigma");

      // ======================================================
      // 1. Sample perturbation directions
      // ======================================================
      final dirs = <_DirectionEval>[];

      for (int d = 0; d < numDirections; d++) {
        final perturb = _RandomPerturb(game.nnue, effSigma);

        final plus = NNUERef.cloneFrom(game.nnue);
        perturb.applyTo(plus, sign: 1);

        final minus = NNUERef.cloneFrom(game.nnue);
        perturb.applyTo(minus, sign: -1);

        double bestPlus = -1e18;
        double bestMinus = -1e18;

        for (final mv in moves) {
          final temp = ChessWithNNUE();
          temp.load(game.fen);
          temp.make_move(mv);

          // +δ
          {
            final acc = plus.newAccumulator();
            plus.refreshAccumulator(acc, temp.board);
            double s = plus.evaluate(acc, temp.turn);
            if (temp.turn == Color.BLACK) s = -s;
            if (s > bestPlus) bestPlus = s;
          }

          // -δ
          {
            final acc = minus.newAccumulator();
            minus.refreshAccumulator(acc, temp.board);
            double s = minus.evaluate(acc, temp.turn);
            if (temp.turn == Color.BLACK) s = -s;
            if (s > bestMinus) bestMinus = s;
          }
        }

        print("      DIR $d: +δ=$bestPlus   -δ=$bestMinus");

        dirs.add(_DirectionEval(perturb, bestPlus, bestMinus));
      }

      dirs.sort((a, b) => max(b.plus, b.minus).compareTo(max(a.plus, a.minus)));

      final bestDirs = dirs.take(topB).toList();
      print("    → Top-$topB directions selected.");

      // ======================================================
      // 2. FINAL MOVE SELECTION USING 2-PLY MINIMAX
      // ======================================================

      Move? bestMove;
      double bestScore = isWhite ? -1e18 : 1e18;

      print("    Evaluating ${moves.length} candidate moves...");

      for (final mv1 in moves) {
        print("      Move1: ${mv1.fromAlgebraic}${mv1.toAlgebraic}");

        // temp1 = board after first move
        final temp1 = ChessWithNNUE();
        temp1.load(game.fen);
        temp1.make_move(mv1);

        final replies = temp1.generate_moves();

        if (replies.isEmpty) {
          final evalTerminal = temp1.nnueEvaluation;
          final score = isWhite ? evalTerminal : -evalTerminal;

          print("         (terminal) score=$score");

          if (isWhite && score > bestScore) {
            bestScore = score;
            bestMove = mv1;
          }
          if (!isWhite && score < bestScore) {
            bestScore = score;
            bestMove = mv1;
          }
          continue;
        }

        // For minimax, opponent selects best reply
        final replyScores = <double>[];

        for (final mv2 in replies) {
          double sum = 0;

          final temp2 = ChessWithNNUE();
          temp2.load(temp1.fen);
          temp2.make_move(mv2);

          for (final dir in bestDirs) {
            final sign = (dir.plus >= dir.minus) ? 1 : -1;

            final clone = NNUERef.cloneFrom(game.nnue);
            dir.pert.applyTo(clone, sign: sign);

            final acc = clone.newAccumulator();
            clone.refreshAccumulator(acc, temp2.board);

            double s = clone.evaluate(acc, temp2.turn);
            if (temp2.turn == Color.BLACK) s = -s;

            sum += s;
          }

          replyScores.add(sum / bestDirs.length);
        }

        // Normalize reply scores
        final normReplies = _normalize(replyScores);

        // Opponent selects best reply
        final opponentBest = (temp1.turn == Color.WHITE)
            ? normReplies.reduce(max)
            : normReplies.reduce(min);

        print("         best OP reply score = $opponentBest");

        if (isWhite) {
          if (opponentBest > bestScore) {
            bestScore = opponentBest;
            bestMove = mv1;
          }
        } else {
          if (opponentBest < bestScore) {
            bestScore = opponentBest;
            bestMove = mv1;
          }
        }
      }

      bestMove ??= moves[rng.nextInt(moves.length)];

      print(
        "    ★ SELECTED MOVE: ${bestMove.fromAlgebraic}${bestMove.toAlgebraic}",
      );
      print("      finalScore=$bestScore\n");

      game.make_move(bestMove);
    }
  }
}

// ================================================================
// Helper classes
// ================================================================
class _DirectionEval {
  final _RandomPerturb pert;
  final double plus;
  final double minus;
  _DirectionEval(this.pert, this.plus, this.minus);
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
    for (int r = 0; r < model.l1.outSize; r++) model.l1.bias[r] += s * l1B[r];

    k = 0;
    for (int c = 0; c < model.l2.inSize; c++) {
      for (int r = 0; r < model.l2.outSize; r++) {
        model.l2.weights[c][r] += s * l2W[k++];
      }
    }
    for (int r = 0; r < model.l2.outSize; r++) model.l2.bias[r] += s * l2B[r];
  }
}
