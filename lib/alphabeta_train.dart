import 'dart:math';
import 'dart:typed_data';
import 'chess3.dart';
import 'chess_nnue2.dart';
import 'nnue_logic_batch.dart';
import 'nnue_persistence.dart';

// --- SEARCH INTEGRATION ---

class SearchResult {
  Move? move;
  double score;
  int nodes;
  List<Move> pv;

  SearchResult(this.move, this.score, {this.nodes = 0, this.pv = const []});
}

class NNUESearcher {
  final ChessWithNNUE game;
  int nodesEvaluated = 0;

  NNUESearcher(this.game);

  SearchResult search(int depth) {
    nodesEvaluated = 0;
    return _alphaBeta(depth, -10000.0, 10000.0, depth);
  }

  SearchResult _alphaBeta(int depth, double alpha, double beta, int maxDepth) {
    nodesEvaluated++;
    if (depth <= 0) return SearchResult(null, game.nnueEvaluation, nodes: 1);

    final moves = game.generate_moves();
    if (moves.isEmpty) {
      if (game.in_check) return SearchResult(null, -999.0 - depth, nodes: 1);
      return SearchResult(null, 0.0, nodes: 1);
    }

    // MVV-LVA Move Ordering
    moves.sort((a, b) {
      int scoreA = a.captured != null ? (a.captured!.shift * 10) : 0;
      int scoreB = b.captured != null ? (b.captured!.shift * 10) : 0;
      return scoreB.compareTo(scoreA);
    });

    Move? bestMove;
    List<Move> bestPv = [];

    for (final move in moves) {
      game.make_move(move);
      var res = _alphaBeta(depth - 1, -beta, -alpha, maxDepth);
      double score = -res.score;
      game.undo_move();

      if (score >= beta) return SearchResult(move, beta, nodes: nodesEvaluated);
      if (score > alpha) {
        alpha = score;
        bestMove = move;
        bestPv = [move, ...res.pv];
      }
    }
    return SearchResult(bestMove, alpha, nodes: nodesEvaluated, pv: bestPv);
  }
}

// --- TRAINING ORCHESTRATOR ---

class NNUETrainer {
  final ChessWithNNUE game;
  final List<TrainingPosition> trainingBuffer = [];
  final int batchSize = 32;

  NNUETrainer(this.game);

  /// Generates training data by searching various positions
  void generateData(int numPositions, int depth) {
    print("Generating $numPositions samples using Depth $depth search...");
    final searcher = NNUESearcher(game);

    for (int i = 0; i < numPositions; i++) {
      // 1. Search the current position to get a "Ground Truth" score
      var res = searcher.search(depth);

      // 2. Add to buffer
      trainingBuffer.add(
        TrainingPosition(List.from(game.board), game.turn, res.score),
      );

      // 3. Make a random move to diversify the training set
      var moves = game.generate_moves();
      if (moves.isNotEmpty) {
        game.make_move(moves[Random().nextInt(moves.length)]);
      } else {
        game.reset(); // If game ends, restart
      }
    }
  }

  /// Runs the Adam Batch Training on the collected data
  void runEpoch(double lr) {
    if (trainingBuffer.length < batchSize) return;

    print("Starting Training Epoch on ${trainingBuffer.length} positions...");
    // Shuffle buffer for better generalization
    trainingBuffer.shuffle();

    // Process in batches
    for (int i = 0; i <= trainingBuffer.length - batchSize; i += batchSize) {
      var batch = trainingBuffer.sublist(i, i + batchSize);
      game.nnue.trainBatch(batch, lr);
    }

    trainingBuffer.clear();
    print("Training Complete. Weights updated via Adam.");
  }
}

// --- MAIN TEST SUITE ---

Future<void> main() async {
  final game = ChessWithNNUE();
  final trainer = NNUETrainer(game);
  final searcher = NNUESearcher(game);

  String modelPath = 'chess_model_v1.json';

  // 1. Load weights if they exist
  await NNUESerializer.load(game, modelPath);

  print("--- [DART-FISH] NNUE TRAINING SYSTEM ---");

  // Step 1: Baseline Evaluation
  var initialRes = searcher.search(3);
  print("Initial Eval: ${initialRes.score.toStringAsFixed(4)}");

  // Step 2: Training Loop
  for (int iteration = 1; iteration <= 5; iteration++) {
    print("\n--- ITERATION $iteration ---");

    // Generate 64 samples via search
    trainer.generateData(64, 2);

    // Train the network on those samples
    trainer.runEpoch(0.001);

    // Step 3: Check progress on the starting position
    game.reset();
    var progressRes = searcher.search(3);
    print("Post-Train Eval (Root): ${progressRes.score.toStringAsFixed(4)}");
  }

  print("\nFinal Test Complete. Engine has 'learned' from search data.");

  // 3. Save progress
  await NNUESerializer.save(myEngine, modelPath);
}
