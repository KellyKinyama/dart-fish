import 'dart:math';
import 'chess3.dart';
// import 'engine3.dart'; // Your class that has zobristKey and make_move
import 'constants.dart';
import 'consts.dart';

final Random _rng = Random();

class MCTSNode {
  final int key; // The Zobrist Key
  int visits = 0;
  double wins = 0;

  // Store moves instead of child node references to handle graph structure
  List<Move>? possibleMoves;
  Map<String, MCTSNode> children = {};

  MCTSNode(this.key);

  bool get isExpanded =>
      possibleMoves != null && children.length == possibleMoves!.length;
}

class MCTS {
  final int iterations;
  final Map<int, MCTSNode> transpositionTable = {};
  final Engine engine;

  MCTS(this.engine, {this.iterations = 1000});

  String? search() {
    transpositionTable.clear();
    final rootKey = engine.zobristKey;
    final root = _getOrCreateNode(rootKey);

    for (int i = 0; i < iterations; i++) {
      _simulateIteration(root);
    }

    // Return the move string ("e2e4") that had the most visits
    String? bestMoveStr;
    int maxVisits = -1;

    root.children.forEach((moveStr, child) {
      if (child.visits > maxVisits) {
        maxVisits = child.visits;
        bestMoveStr = moveStr;
      }
    });

    return bestMoveStr;
  }

  /// This replaces the broken _uctSelect.
  /// It picks the move that maximizes the UCT score.
  // Move _uctSelectMove(MCTSNode node) {
  //   const double c = 1.414; // Exploration parameter
  //   Move? bestMove;
  //   double bestValue = -double.infinity;

  //   // We iterate through the children mapped by move strings
  //   node.children.forEach((moveStr, childNode) {
  //     double exploitation = childNode.wins / (childNode.visits + 1e-6);
  //     double exploration = sqrt(
  //       log(node.visits + 1) / (childNode.visits + 1e-6),
  //     );

  //     // Note: In MCTS, wins are usually relative to the player who just moved
  //     double uctValue = exploitation + c * exploration;

  //     if (uctValue > bestValue) {
  //       bestValue = uctValue;
  //       // Map the string back to the engine's Move object
  //       bestMove = node.possibleMoves!.firstWhere(
  //         (m) => (m.fromAlgebraic + m.toAlgebraic) == moveStr,
  //       );
  //     }
  //   });

  //   return bestMove!;
  // }

  /// Picking the move using the requested formula: Q + C * P * sqrt(N) / (1 + n)
  Move _uctSelectMove(MCTSNode node) {
    const double C = 1.414;
    Move? bestMove;
    double bestValue = -double.infinity;

    final double N = node.visits.toDouble();

    node.children.forEach((moveStr, childNode) {
      // 1. Calculate Q (Exploitation)
      double Q = childNode.wins / (childNode.visits + 1e-6);

      // 2. Get P (Prior Probability/Heuristic)
      // We find the move object to determine its "Urgency"
      final moveObj = node.possibleMoves!.firstWhere(
        (m) => (m.fromAlgebraic + m.toAlgebraic) == moveStr,
      );
      double P = _getP(moveObj);

      // 3. Calculate n (Child visits)
      double n = childNode.visits.toDouble();

      // 4. Apply the Formula: Q + C * P * sqrt(N) / (1 + n)
      double uctValue = Q + (C * P * sqrt(N) / (1 + n));

      if (uctValue > bestValue) {
        bestValue = uctValue;
        bestMove = moveObj;
      }
    });

    return bestMove!;
  }

  /// Assigns a "Prior" value to moves so the search explores smart moves first.
  /// This acts as the 'P' in your formula.
  double _getP(Move move) {
    double priority = 1.0; // Base priority for a "Normal" move

    // Give priority to captures
    if ((move.flags & BITS_CAPTURE) != 0) priority += 0.5;

    // Give priority to promotions
    if ((move.flags & BITS_PROMOTION) != 0) priority += 0.8;

    // Give priority to big pawn pushes (controlling center)
    if ((move.flags & BITS_BIG_PAWN) != 0) priority += 0.2;

    return priority;
  }

  void _simulateIteration(MCTSNode root) {
    List<MCTSNode> path = [root];
    List<Move> moveHistory = [];

    // 1. Selection
    MCTSNode current = root;
    while (current.isExpanded) {
      Move move = _uctSelectMove(current);
      engine.make_move(move);
      moveHistory.add(move);

      current = _getOrCreateNode(engine.zobristKey);
      path.add(current);
      if (engine.game_over) break;
    }

    // 2. Expansion
    if (!engine.game_over) {
      current.possibleMoves ??= engine.generate_moves();
      for (var move in current.possibleMoves!) {
        String moveStr = move.fromAlgebraic + move.toAlgebraic;
        if (!current.children.containsKey(moveStr)) {
          engine.make_move(move);
          MCTSNode childNode = _getOrCreateNode(engine.zobristKey);
          current.children[moveStr] = childNode;
          path.add(childNode);
          moveHistory.add(move);
          break;
        }
      }
    }

    // 3. Simulation (Rollout)
    double result = _rollout();

    // 4. Backpropagate (and undo moves to reset engine)
    for (int i = path.length - 1; i >= 0; i--) {
      path[i].visits++;
      path[i].wins += result;
      result = -result; // Flip perspective for opponent
      if (i > 0) engine.undo_move();
    }
  }

  MCTSNode _getOrCreateNode(int key) {
    return transpositionTable.putIfAbsent(key, () => MCTSNode(key));
  }

  // Move _uctSelectMove(MCTSNode node) {
  //   const double c = 1.41;
  //   Move? bestMove;
  //   double bestValue = -double.infinity;

  //   node.children.forEach((moveStr, child) {
  //     double exploitation = child.wins / (child.visits + 1e-6);
  //     double exploration = sqrt(log(node.visits + 1) / (child.visits + 1e-6));
  //     double uct = exploitation + c * exploration;

  //     if (uct > bestValue) {
  //       bestValue = uct;
  //       // Find the move object corresponding to the move string
  //       bestMove = node.possibleMoves!.firstWhere(
  //         (m) => (m.fromAlgebraic + m.toAlgebraic) == moveStr,
  //       );
  //     }
  //   });
  //   return bestMove!;
  // }

  double _rollout() {
    int depth = 0;
    final List<Move> rolledMoves = [];

    while (!engine.game_over && depth < 30) {
      final moves = engine.generate_moves();
      if (moves.isEmpty) break;
      final m = moves[_rng.nextInt(moves.length)];
      engine.make_move(m);
      rolledMoves.add(m);
      depth++;
    }

    double score = 0;
    if (engine.in_checkmate) {
      score = -1.0; // Current player lost
    } else {
      // Basic heuristic for non-terminal rollouts
      score = engine.eval() / 1000.0;
      score = score.clamp(-0.9, 0.9);
    }

    // Cleanup simulation moves
    for (int i = 0; i < rolledMoves.length; i++) {
      engine.undo_move();
    }
    return score;
  }
}

class Engine extends Chess {
  Engine() : super();
  Engine.fromFEN(super.fen) : super.fromFEN();

  // Helper to convert 0x88 index to 0-63
  int _to63(int i) => (i >> 4) * 8 + (i & 7);

  double eval() {
    if (insufficient_material) return 0;
    if (in_stalemate || in_threefold_repetition) return -500;
    if (in_checkmate)
      return -9999999; // Return negative because it's usually bad for the person moving

    double score = 0;

    // We only loop through valid squares (0x88 board)
    for (int i = 0; i < 128; i++) {
      if ((i & 0x88) != 0) continue;

      final piece = board[i];
      if (piece is Piece) {
        final pType = piece.type.toString();
        final material = material_values[pType] ?? 0;

        // Side multiplier: White 1, Black -1
        final sideMultiplier = (piece.color == Chess.WHITE) ? 1 : -1;
        score += material * sideMultiplier;

        // Piece Square Tables
        final table = PieceSquareTables[pType];
        if (table != null) {
          int index = (piece.color == Chess.WHITE) ? _to63(i) : 63 - _to63(i);
          score += (table[index] ?? 0.0) * sideMultiplier;
        }
      }
    }

    // Adjust score based on whose turn it is
    return (turn == Chess.WHITE) ? score : -score;
  }

  /// Holds the node count of the most recent search for debugging purposes.
  int lastSearchNodeCount = 0;

  /// Executes a Graph-based MCTS search and applies the best move.
  String play({int iterations = 1000}) {
    // 1. Initialize MCTS with this engine instance.
    // This allows MCTS to use our Zobrist keys and incremental move logic.
    final mcts = MCTS(this, iterations: iterations);

    // 2. Perform the search.
    // The search method returns the best move in LAN format (e.g., "e2e4").
    final String? bestMoveLAN = mcts.search();

    // 3. Update debug stats.
    lastSearchNodeCount = mcts.transpositionTable.length;

    if (bestMoveLAN == null || bestMoveLAN.isEmpty) {
      return "";
    }

    // 4. Apply the move to our own internal state.
    // We use moveLAN to handle standard moves, castling, and promotions.
    bool success = moveLAN(bestMoveLAN);

    if (!success) {
      print("CRITICAL: MCTS suggested an invalid move: $bestMoveLAN");
      return "";
    }

    return bestMoveLAN;
  }

  /// Handles Long Algebraic Notation (LAN) moves.
  /// Converts strings like "e2e4" or "e1g1" into internal engine moves.
  bool moveLAN(String lanstr) {
    final e1 = get('e1');

    if (lanstr == "e1g1" && e1 is Piece && e1.type.name == 'k') {
      return move("O-O");
    }

    if (lanstr == "e1c1" && e1 is Piece && e1.type.name == 'k') {
      return move("O-O-O");
    }

    var coords = {'from': lanstr.substring(0, 2), 'to': lanstr.substring(2, 4)};

    if (lanstr.length > 4) {
      coords['promotion'] = lanstr[4].toLowerCase();
    }

    return move(coords);
  }

  /// Helper to convert our internal Move objects back to LAN strings if needed.
  // String moveToString(Move move) {
  //   String res = move.fromAlgebraic + move.toAlgebraic;
  //   if ((move.flags & BITS_PROMOTION) != 0) {
  //     res += move.promotion!.name.toLowerCase();
  //   }
  //   return res;
  // }
}
