import 'package:dart_fish/alphabeta_train2.dart';
import 'package:dart_fish/chess_nnue3.dart';
import 'package:dart_fish/nnue_persistence2.dart';

Future<void> main() async {
  final game = ChessWithNNUE();
  final trainer = NNUETrainer(game);
  final searcher = NNUESearcher(game);

  const String modelPath = 'chess_model_v2.json';

  // 1) Load weights if available
  try {
    await NNUESerializer.load(game.nnue, modelPath);
    print("Loaded NNUE model from $modelPath");
  } catch (e) {
    print("No existing model to load (or load failed): $e");
  }

  print("--- [DART-FISH] NNUE TRAINING SYSTEM ---");

  // Baseline
  final initialRes = searcher.search(3);
  print(
    "Initial Eval: ${initialRes.score.toStringAsFixed(4)} cp | nodes: ${initialRes.nodes}",
  );

  // Training loop
  const int iterations = 200;
  const int samplesPerIter = 64;
  const int labelDepth = 2;
  const double lr = 0.001;

  for (int iteration = 1; iteration <= iterations; iteration++) {
    print("\n--- ITERATION $iteration ---");

    trainer.generateData(samplesPerIter, labelDepth);
    trainer.runEpoch(lr);

    // Check progress from starting position
    game.reset();
    final progressRes = searcher.search(3);
    print(
      "Post-Train Eval (Root): ${progressRes.score.toStringAsFixed(4)} cp | nodes: ${progressRes.nodes}",
    );
  }

  print("\nFinal Test Complete. Engine has 'learned' from search data.");

  // 3) Save progress (FIXED: pass `game.nnue`, not `game`)
  try {
    await NNUESerializer.save(game.nnue, modelPath);
    print("Saved NNUE model to $modelPath");
  } catch (e) {
    print("Failed to save model: $e");
  }
}
