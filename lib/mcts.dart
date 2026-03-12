import 'dart:math';
import "package:chess/chess.dart";
import 'consts.dart';

final Random _rng = Random();

class MCTSNode {
  Chess state;
  MCTSNode? parent;
  String? move;

  List<MCTSNode> children = [];

  int visits = 0;
  double wins = 0;

  MCTSNode(this.state, {this.parent, this.move});

  bool get isFullyExpanded {
    return children.length == state.moves().length;
  }
}

class MCTS {
  int iterations;

  MCTS({this.iterations = 500});

  String? search(String fen) {
    final root = MCTSNode(Chess.fromFEN(fen));

    for (int i = 0; i < iterations; i++) {
      MCTSNode node = _select(root);
      node = _expand(node);

      double result = _simulate(node.state);

      _backpropagate(node, result);
    }

    MCTSNode? best;
    int maxVisits = -1;

    for (final child in root.children) {
      if (child.visits > maxVisits) {
        maxVisits = child.visits;
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
    double bestValue = -double.infinity;

    for (final child in node.children) {
      double exploitation = child.wins / (child.visits + 1e-6);
      double exploration = sqrt(log(node.visits + 1) / (child.visits + 1e-6));

      double uct = exploitation + c * exploration;

      if (uct > bestValue) {
        bestValue = uct;
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
        final newState = Chess.fromFEN(node.state.fen);
        newState.move(move);

        final child = MCTSNode(newState, parent: node, move: move);

        node.children.add(child);
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
    while (node.parent != null) {
      node.visits += 1;
      node.wins += result;

      node = node.parent!;
      result = -result;
    }

    node.visits += 1;
    node.wins += result;
  }
}

class Engine extends Chess {
  Engine() : super();
  Engine.fromFEN(super.fen) : super.fromFEN();

  String play() {
    final mcts = MCTS(iterations: 1000);

    final bestMove = mcts.search(fen);

    if (bestMove == null) return "";

    move(bestMove);

    var lastMove = getHistory({"verbose": true}).last;

    String lanstr = lastMove['from'] + lastMove['to'];

    if (lastMove['flags'].contains('p')) {
      String san = lastMove['san'].toString().toLowerCase();
      lanstr += san.substring(san.length - 1);
    }

    return lanstr;
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
