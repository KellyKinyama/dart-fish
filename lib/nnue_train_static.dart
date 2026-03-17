// lib/nnue_train_static.dart
import 'dart:math';
import 'chess_with_nnue.dart';
import 'nnue_reference.dart';
import 'nnue_logic_batch2.dart';
import 'nnue_persistence_ref.dart';
import 'nnue_static_dataset5.dart';

Future<void> main() async {
  final game = ChessWithNNUE();
  final datasetBuilder = StaticDatasetBuilder();

  const modelPath = "nnue_static_model.json";

  // Try to load existing model
  final ok = await NNUESerializer.load(game.nnue, modelPath);
  print(ok ? "Loaded NNUE model from $modelPath" : "Start random NNUE.");

  // Hyperparameters (tune as needed)
  const int iterations = 1000;
  const int samplesPerIter = 64;
  const int batchSize = 16;
  const double lr = 0.0005;

  print("--- STATIC NNUE TRAINING ---");

  for (int it = 1; it <= iterations; it++) {
    print("\n=== ITERATION $it/$iterations ===");

    // Build labeled dataset using static eval (independent teacher)
    final samples = datasetBuilder.generate(
      game,
      numPositions: samplesPerIter,
      playoutDepth: 4,
    );

    // Shuffle and mini-batch train
    samples.shuffle(Random());

    for (int b = 0; b <= samples.length - batchSize; b += batchSize) {
      final batch = samples.sublist(b, b + batchSize);

      final pre = _batchMseNet(game.nnue, batch);
      game.nnue.trainBatch(batch, lr);
      final post = _batchMseNet(game.nnue, batch);

      print(
        "  batch ${b ~/ batchSize}: "
        "preNet=${pre.toStringAsFixed(6)}  "
        "postNet=${post.toStringAsFixed(6)}",
      );
    }

    // Small validation set
    final valset = datasetBuilder.generate(
      game,
      numPositions: 64,
      playoutDepth: 4,
    );
    final valMSE = _batchMseNet(game.nnue, valset);
    print("Validation MSE (net): ${valMSE.toStringAsFixed(6)}");

    // Save
    // if (it % 200 == 0) {
    await NNUESerializer.save(game.nnue, modelPath);

    print("Saved -> $modelPath");
    // }
  }
}

double _batchMseNet(NNUERef model, List<TrainingPosition> batch) {
  double sse = 0.0;
  for (final tp in batch) {
    final predCp = model.evalBoardCp(tp.board, tp.turn);
    final pred = predCp / NNUERef.SCALE_CP;
    final target = tp.target / NNUERef.SCALE_CP;
    final diff = pred - target;
    sse += diff * diff;
  }
  return sse / (batch.isEmpty ? 1 : batch.length);
}
