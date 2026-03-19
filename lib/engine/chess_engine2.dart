// ---------------------------------------------------------
//  FULL CHESS ENGINE WITH DEBUGGING:
//  • NNUE VALUE HEAD
//  • TRANSFORMER POLICY HEAD (AFT Transformer)
//  • ALPHAZERO-STYLE MCTS WITH PUCT
//  • EXTENSIVE DEBUG LOGGING
// ---------------------------------------------------------

import 'dart:math';
import 'chess.dart';
import 'constants.dart';
import 'chess_with_nnue.dart';
import 'nnue_persistence_ref.dart';
import 'package:dart_cuda/gpu_tensor.dart';
import 'package:dart_cuda/aft_transformer_decoder.dart';
import 'package:dart_cuda/network_utils.dart';

// ---------------------------------------------------------
// MOVE ENCODING FOR POLICY TRANSFORMER
// ---------------------------------------------------------

int encodeMove(String uci) {
  if (uci == "<start>") return 4096;
  if (uci == ".") return 4097;

  int sqToIdx(String sq) {
    int file = sq.codeUnitAt(0) - 'a'.codeUnitAt(0);
    int rank = int.parse(sq[1]) - 1;
    return rank * 8 + file;
  }

  return (sqToIdx(uci.substring(0, 2)) * 64) + sqToIdx(uci.substring(2, 4));
}

String decodeMove(int index) {
  if (index == 4096) return "<start>";
  if (index == 4097) return ".";

  String idxToSq(int idx) {
    return String.fromCharCode('a'.codeUnitAt(0) + (idx % 8)) +
        (idx ~/ 8 + 1).toString();
  }

  return idxToSq(index ~/ 64) + idxToSq(index % 64);
}

class _DebugMove {
  final String lan;
  final double q;
  final double p;
  final double n;
  final double score;

  _DebugMove(this.lan, this.q, this.p, this.n, this.score);
}

// ---------------------------------------------------------
//  MCTS NODE
// ---------------------------------------------------------

class MCTSNode {
  final int key;
  int visits = 0;
  double wins = 0.0;

  List<Move>? possibleMoves;
  final Map<String, MCTSNode> children = {};

  Map<String, double>? policyPriors;

  bool get isExpanded =>
      possibleMoves != null && children.length == possibleMoves!.length;

  MCTSNode(this.key);
}

// ---------------------------------------------------------
//  MCTS WITH TRANSFORMER POLICY + NNUE VALUE
// ---------------------------------------------------------

class MCTS {
  final ChessWithNNUE game;

  final TransformerDecoder policyNet;
  final Tensor dummyEnc;
  final int blockSize;

  final int iterations;
  final double cPuct;

  final Map<int, MCTSNode> transpositionTable = {};

  MCTS(
    this.game,
    this.policyNet,
    this.dummyEnc,
    this.blockSize, {
    this.iterations = 2000,
    this.cPuct = 1.2,
  });

  // ---------------------------------------------------------
  //  SEARCH ENTRYPOINT
  // ---------------------------------------------------------
  String? search() {
    transpositionTable.clear();

    print("[DEBUG] Starting search at Zobrist=${game.zobristKey}");

    final rootKey = game.zobristKey;
    final root = _getOrCreateNode(rootKey);

    for (int i = 0; i < iterations; i++) {
      if (i % 500 == 0) {
        print("[DEBUG] Iteration $i / $iterations");
      }
      _simulate(root);
    }

    print("[DEBUG] SEARCH DONE. ROOT CHILDREN EVALUATION:");

    String? bestMove;
    int maxVisits = -1;

    root.children.forEach((lan, node) {
      print("[DEBUG] Move $lan -> visits=${node.visits}, wins=${node.wins}");
      if (node.visits > maxVisits) {
        maxVisits = node.visits;
        bestMove = lan;
      }
    });

    print("[DEBUG] BEST ROOT MOVE = $bestMove");
    return bestMove;
  }

  // ---------------------------------------------------------
  //  MONTE CARLO TREE SEARCH
  // ---------------------------------------------------------
  void _simulate(MCTSNode root) {
    final List<MCTSNode> path = [root];
    final List<Move> played = [];

    MCTSNode node = root;

    // -------------------------
    // 1. SELECTION
    // -------------------------
    while (node.isExpanded && !game.game_over) {
      Move move = _selectMove(node);
      print("[DEBUG] SELECT -> ${_lanOf(move)}");

      game.make_move(move);
      played.add(move);

      node = _getOrCreateNode(game.zobristKey);
      path.add(node);
    }

    // -------------------------
    // 2. EXPANSION
    // -------------------------
    if (!game.game_over) {
      node.possibleMoves ??= game.generate_moves({'legal': true});
      print("[DEBUG] EXPANDING NODE: ${node.possibleMoves!.length} moves");

      if (node.policyPriors == null) {
        print("[DEBUG] Computing Transformer priors...");
        node.policyPriors = _computePolicyPriors(node);
      }

      for (final m in node.possibleMoves!) {
        final lan = _lanOf(m);
        if (!node.children.containsKey(lan)) {
          print("[DEBUG] EXPAND CHILD = $lan");

          game.make_move(m);
          played.add(m);

          final child = _getOrCreateNode(game.zobristKey);
          node.children[lan] = child;
          path.add(child);
          break;
        }
      }
    }

    // -------------------------
    // 3. EVALUATION
    // -------------------------
    final value = _evaluateLeaf();
    print("[DEBUG] LEAF VALUE = $value");

    // -------------------------
    // 4. BACKPROPAGATION
    // -------------------------
    double v = value;
    for (int i = path.length - 1; i >= 0; i--) {
      final n = path[i];
      n.visits++;
      n.wins += v;
      print("[DEBUG] BP node idx=$i visits=${n.visits} wins=${n.wins}");
      v = -v;

      if (i > 0) game.undo_move();
    }
  }

  // ---------------------------------------------------------
  //  PUCT MOVE SELECTION
  // ---------------------------------------------------------
  Move _selectMove(MCTSNode node) {
    node.possibleMoves ??= game.generate_moves({'legal': true});

    final List<_DebugMove> dbg = [];

    Move? best;
    double bestScore = -999999;

    final double N = (node.visits + 1).toDouble();

    print("[DEBUG] COMPUTING UCT SCORES (nodeVisits=$N)");

    for (final m in node.possibleMoves!) {
      final lan = _lanOf(m);
      final child = node.children[lan];

      final double q = (child == null || child.visits == 0)
          ? 0.0
          : child.wins / child.visits;

      final double p = node.policyPriors?[lan] ?? 1e-6;
      final double n = (child?.visits ?? 0).toDouble();

      final double score = q + cPuct * p * sqrt(N) / (1 + n);

      dbg.add(_DebugMove(lan, q, p, n, score));

      if (score > bestScore) {
        bestScore = score;
        best = m;
      }
    }

    // -----------------------------
    // SORT BY POLICY PRIOR
    // -----------------------------
    dbg.sort((a, b) => b.p.compareTo(a.p));

    print("[DEBUG] ---- SORTED BY POLICY p ----");
    for (final d in dbg) {
      print("[DEBUG] ${d.lan}: p=${d.p} q=${d.q} n=${d.n} uctScore=${d.score}");
    }
    print("[DEBUG] --------------------------------");

    print("[DEBUG] SELECTED MOVE = ${_lanOf(best!)}");
    return best!;
  }

  // ---------------------------------------------------------
  //  NNUE VALUE HEAD
  // ---------------------------------------------------------
  double _evaluateLeaf() {
    if (game.in_checkmate) {
      print("[DEBUG] EvaluateLeaf -> Checkmate = -1");
      return -1.0;
    }
    if (game.in_stalemate ||
        game.insufficient_material ||
        game.in_threefold_repetition) {
      print("[DEBUG] EvaluateLeaf -> Draw = 0");
      return 0.0;
    }

    final cp = game.nnueEvaluation;
    final norm = (cp / 1000).clamp(-1.0, 1.0);

    print("[DEBUG] NNUE CP=$cp  normalized=$norm");
    return norm;
  }

  // ---------------------------------------------------------
  //  POLICY HEAD (TRANSFORMER)
  // ---------------------------------------------------------
  Map<String, double> _computePolicyPriors(MCTSNode node) {
    final moves = node.possibleMoves!;

    final List<int> historyIds = [];

    print("[DEBUG] Building history for Transformer...");

    for (final s in game.history) {
      final mv = s.move;
      if (mv == null) continue;
      final lan = _lanOf(mv);

      print("[DEBUG] History move: $lan");

      historyIds.add(encodeMove(lan));
    }

    // -------------------------
    // FIX: SAFE CONTEXT
    // -------------------------
    List<int> context;

    if (historyIds.isEmpty) {
      print("[DEBUG] No history -> using <start> token");
      context = [encodeMove("<start>")];
    } else {
      context = historyIds.length > blockSize
          ? historyIds.sublist(historyIds.length - blockSize)
          : historyIds;
    }

    print("[DEBUG] Transformer context length = ${context.length}");
    print("[DEBUG] Context tokens = $context");

    // Forward Transformer
    final tracker = <Tensor>[];
    final logits = policyNet.forward(context, dummyEnc, tracker);

    final row = logits.fetchRow(context.length - 1);

    // Softmax
    final maxLog = row.reduce(max);
    final exps = row.map((v) => exp(v - maxLog)).toList();
    final sum = exps.reduce((a, b) => a + b);
    final probs = exps.map((e) => e / sum).toList();

    print("[DEBUG] Transformer priors computed. Assigning...");

    final Map<String, double> priors = {};

    for (final m in moves) {
      final lan = _lanOf(m);
      final id = encodeMove(lan);
      priors[lan] = probs[id];

      print("[DEBUG] PRIOR $lan = ${priors[lan]}");
    }

    for (var t in tracker) t.dispose();
    logits.dispose();

    return priors;
  }

  // ---------------------------------------------------------
  //  HELPERS
  // ---------------------------------------------------------

  MCTSNode _getOrCreateNode(int key) =>
      transpositionTable.putIfAbsent(key, () => MCTSNode(key));

  String _lanOf(Move m) {
    var lan = m.fromAlgebraic + m.toAlgebraic;
    if ((m.flags & BITS_PROMOTION) != 0 && m.promotion != null) {
      lan += m.promotion!.name.toLowerCase();
    }
    return lan;
  }
}

// ---------------------------------------------------------
//  ENGINE MAIN ENTRYPOINT
// ---------------------------------------------------------

Future<void> main() async {
  final game = ChessWithNNUE();

  const String modelPath = "chess_model_v1.json";
  const String weightPath = "chess_gpt.bin";

  print("[DEBUG] Loading NNUE...");
  try {
    await NNUESerializer.load(game.nnue, modelPath);
    print("[DEBUG] Loaded NNUE OK");
  } catch (e) {
    print("[DEBUG] NNUE load error: $e");
  }

  print("[DEBUG] Initializing Transformer...");
  const vocabSize = 4098;
  const bigSize = 16;
  const blockSize = 16;

  final policyNet = TransformerDecoder(
    vocabSize: vocabSize,
    embedSize: bigSize,
    encoderEmbedSize: bigSize,
    numLayers: 2,
    numHeads: 4,
    blockSize: blockSize,
  );

  final dummyEnc = Tensor.zeros([1, bigSize]);

  print("[DEBUG] Loading Transformer weights...");
  await loadModuleBinary(policyNet, weightPath);

  print("[DEBUG] Building MCTS...");
  final mcts = MCTS(
    game,
    policyNet,
    dummyEnc,
    blockSize,
    iterations: 4000,
    cPuct: 1.2,
  );

  print("[DEBUG] Starting search...");
  final bestMove = mcts.search();

  if (bestMove == null) {
    print("No legal moves (game over).");
  } else {
    print("[DEBUG] ENGINE PLAYS: $bestMove");
    game.moveLAN(bestMove);
  }
}
