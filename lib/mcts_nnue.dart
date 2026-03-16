import 'dart:math';
import 'chess3.dart';
import 'constants.dart';
import 'chess_nnue3.dart';
import 'nnue_persistence2.dart'; // Provides ChessWithNNUE with nnueEvaluation

// final Random _rng = Random();

/// A node is keyed by the Zobrist hash of a position.
/// Children are addressed by LAN move strings (e.g., "e2e4", "e7e8q").
class MCTSNode {
  final int key; // Zobrist Key
  int visits = 0;
  double wins =
      0.0; // Sum of results (+1 win, -1 loss, 0 draw) from this node's POV

  List<Move>? possibleMoves; // Cached legal moves for the node
  final Map<String, MCTSNode> children = {}; // LAN -> child node

  MCTSNode(this.key);

  bool get isExpanded =>
      possibleMoves != null && children.length == possibleMoves!.length;
}

class MCTS {
  final ChessWithNNUE game;
  final int iterations;
  final Map<int, MCTSNode> transpositionTable = {};

  // Exploration weight (PUCT-style)
  final double cPuct;

  MCTS(this.game, {this.iterations = 1000, this.cPuct = 1.414});

  /// Run MCTS from the current game position and return best move in LAN ("e2e4", "e7e8q", ...).
  String? search() {
    transpositionTable.clear();

    final rootKey = game.zobristKey;
    final root = _getOrCreateNode(rootKey);

    for (int i = 0; i < iterations; i++) {
      _simulate(root);
    }

    // Pick child with highest visits
    String? bestMoveStr;
    int maxVisits = -1;

    root.children.forEach((lan, child) {
      if (child.visits > maxVisits) {
        maxVisits = child.visits;
        bestMoveStr = lan;
      }
    });

    return bestMoveStr;
  }

  // --- Core MCTS phases ---

  void _simulate(MCTSNode root) {
    final List<MCTSNode> path = [root];
    final List<Move> playedMoves = [];

    // 1) Selection: Follow UCT until we reach an unexpanded node or terminal.
    MCTSNode node = root;
    while (node.isExpanded && !game.game_over) {
      final Move move = _selectMove(node);
      game.make_move(move);
      playedMoves.add(move);

      node = _getOrCreateNode(game.zobristKey);
      path.add(node);
    }

    // 2) Expansion: Expand one new child (if not terminal).
    if (!game.game_over) {
      node.possibleMoves ??= game.generate_moves();
      if (node.possibleMoves!.isEmpty) {
        // No legal moves (mate or stalemate): no expansion
      } else {
        // Add first unseen move as a new child
        for (final m in node.possibleMoves!) {
          final lan = _lanOf(m);
          if (!node.children.containsKey(lan)) {
            game.make_move(m);
            playedMoves.add(m);

            final child = _getOrCreateNode(game.zobristKey);
            node.children[lan] = child;
            path.add(child);

            break; // Expand one child per simulation
          }
        }
      }
    }

    // 3) Evaluation (no random rollout): use NNUE value at current game state.
    final double result = _evaluateLeaf();

    // 4) Backpropagate result and undo played moves.
    double value = result;
    for (int i = path.length - 1; i >= 0; i--) {
      final n = path[i];
      n.visits++;
      n.wins += value; // accumulate value from node's perspective
      value = -value; // flip perspective at each ply

      if (i > 0) game.undo_move(); // undo the move that led to this node
    }
  }

  // --- Selection ---

  Move _selectMove(MCTSNode node) {
    // Ensure we have moves generated and at least one child to select
    node.possibleMoves ??= game.generate_moves();

    Move? bestMove;
    double bestScore = -double.infinity;

    final double N = node.visits.toDouble();

    for (final m in node.possibleMoves!) {
      final lan = _lanOf(m);
      final child = node.children[lan];

      // Q = mean value; if unvisited, treat as 0 for exploitation
      final double q = (child == null || child.visits == 0)
          ? 0.0
          : (child.wins / child.visits);

      // P = prior (simple heuristic)
      final double p = _prior(m);

      final double n = (child?.visits ?? 0).toDouble();
      final double uct =
          q + cPuct * p * (sqrt(N + 1e-9) / (1.0 + n)); // +1e-9 for stability

      if (uct > bestScore) {
        bestScore = uct;
        bestMove = m;
      }
    }

    return bestMove!;
  }

  // --- Evaluation ---

  /// NNUE value in [-1, 1] from the side-to-move perspective at the leaf.
  double _evaluateLeaf() {
    if (game.in_checkmate) {
      return -1.0; // side to move is checkmated
    }
    if (game.in_stalemate ||
        game.insufficient_material ||
        game.in_threefold_repetition) {
      return 0.0; // draw
    }

    // NNUE returns centipawns from side-to-move perspective
    final cp = game.nnueEvaluation;

    // Normalize to [-1, 1] (tunable). 1000 = 10 pawns. You can choose 800 or 1200 as well.
    final v = (cp / 1000.0).clamp(-1.0, 1.0);
    return v;
    // If you want slight optimism for unvisited nodes, you could add a small epsilon here.
  }

  // --- Helpers ---

  MCTSNode _getOrCreateNode(int key) =>
      transpositionTable.putIfAbsent(key, () => MCTSNode(key));

  /// Simple, fast prior: emphasize captures, promotions, and big pawn pushes.
  double _prior(Move move) {
    double p = 1.0; // base
    if ((move.flags & BITS_CAPTURE) != 0) p += 0.5;
    if ((move.flags & BITS_PROMOTION) != 0) p += 0.8;
    if ((move.flags & BITS_BIG_PAWN) != 0) p += 0.2;
    return p;
  }

  /// LAN string for a move: "e2e4", "e7e8q", etc.
  String _lanOf(Move m) {
    var s = m.fromAlgebraic + m.toAlgebraic;
    if ((m.flags & BITS_PROMOTION) != 0 && m.promotion != null) {
      s += m.promotion!.name.toLowerCase();
    }
    return s;
  }
}

Future<void> main() async {
  final game = ChessWithNNUE();

  // Optional: load weights if you have a serializer
  // await NNUESerializer.load(game, 'chess_model_v1.json');

  const String modelPath = 'chess_model_v1.json';

  // 1) Load weights if available
  try {
    await NNUESerializer.load(game.nnue, modelPath);
    print("Loaded NNUE model from $modelPath");
  } catch (e) {
    print("No existing model to load (or load failed): $e");
  }

  final mcts = MCTS(game, iterations: 4000, cPuct: 1.2);

  // Best move as LAN
  final bestLan = mcts.search();
  if (bestLan != null) {
    print('MCTS suggests: $bestLan');
    game.moveLAN(bestLan); // implement or reuse your own LAN applier
  } else {
    print('No legal moves found (game over).');
  }
}
