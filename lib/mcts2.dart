import 'dart:math';
import 'package:chess/chess.dart';

final Random _rng = Random();

/// Transposition table
final Map<String, MCTSNode> transTable = {};

class MCTSNode {
  Chess state;
  String? move;

  List<MCTSNode> children = [];
  List<MCTSNode> parents = [];

  int visits = 0;
  double wins = 0;

  MCTSNode(this.state, {this.move});

  bool get isFullyExpanded {
    return children.length == state.moves().length;
  }
}

class MCTS {
  int iterations;

  MCTS({this.iterations = 1000});

  String? search(String fen) {
    final root = transTable.putIfAbsent(
      fen,
      () => MCTSNode(Chess.fromFEN(fen)),
    );

    for (int i = 0; i < iterations; i++) {
      MCTSNode node = _select(root);
      node = _expand(node);

      double result = _simulate(node.state);

      _backpropagate(node, result);
    }

    MCTSNode? best;
    int bestVisits = -1;

    for (final child in root.children) {
      if (child.visits > bestVisits) {
        bestVisits = child.visits;
        best = child;
      }
    }

    return best?.move;
  }

  MCTSNode _select(MCTSNode node) {
    while (node.children.isNotEmpty && node.isFullyExpanded) {
      node = _uct(node);
    }
    return node;
  }

  MCTSNode _uct(MCTSNode node) {
    const double c = 1.41;

    MCTSNode? best;
    double bestScore = -double.infinity;

    for (final child in node.children) {
      double exploitation = child.wins / (child.visits + 1e-6);
      double exploration = sqrt(log(node.visits + 1) / (child.visits + 1e-6));

      double uct = exploitation + c * exploration;

      if (uct > bestScore) {
        bestScore = uct;
        best = child;
      }
    }

    return best!;
  }

  MCTSNode _expand(MCTSNode node) {
    final moves = node.state.moves();
    final triedMoves = node.children.map((c) => c.move).toSet();

    for (final move in moves) {
      if (!triedMoves.contains(move)) {
        final next = Chess.fromFEN(node.state.fen);
        next.move(move);

        final key = next.fen;

        // Transposition check
        if (transTable.containsKey(key)) {
          final existing = transTable[key]!;

          node.children.add(existing);
          existing.parents.add(node);

          return existing;
        }

        final child = MCTSNode(next, move: move);

        node.children.add(child);
        child.parents.add(node);

        transTable[key] = child;

        return child;
      }
    }

    return node;
  }

  double _simulate(Chess board) {
    Chess sim = Chess.fromFEN(board.fen);

    int rolloutDepth = 40;

    while (!sim.game_over && rolloutDepth-- > 0) {
      final moves = sim.moves();

      if (moves.isEmpty) break;

      final move = moves[_rng.nextInt(moves.length)];
      sim.move(move);
    }

    if (sim.in_checkmate) {
      return sim.turn == Color.WHITE ? -1 : 1;
    }

    return 0;
  }

  void _backpropagate(MCTSNode node, double result) {
    final stack = <MCTSNode>[node];
    final visited = <MCTSNode>{};

    while (stack.isNotEmpty) {
      final current = stack.removeLast();

      if (!visited.add(current)) continue;

      current.visits++;
      current.wins += result;

      for (final parent in current.parents) {
        stack.add(parent);
      }

      result = -result;
    }
  }
}

class Engine extends Chess {
  Engine() : super();
  Engine.fromFEN(super.fen) : super.fromFEN();

  String play() {
    final mcts = MCTS(iterations: 2000);

    final bestMove = mcts.search(fen);

    if (bestMove == null) return "";

    move(bestMove);

    final lastMove = getHistory({"verbose": true}).last;

    String lan = lastMove['from'] + lastMove['to'];

    if (lastMove['flags'].contains('p')) {
      String san = lastMove['san'].toString().toLowerCase();
      lan += san.substring(san.length - 1);
    }

    return lan;
  }

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
}

void main() {
  Engine engine = Engine();

  print("Initial Position:");
  print(engine.ascii);

  for (int i = 0; i < 10; i++) {
    String move = engine.play();

    if (move.isEmpty) {
      print("Game over");
      break;
    }

    print("Move: $move");
    print(engine.ascii);
  }
}
