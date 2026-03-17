import 'dart:io';
import 'chess3.dart';
import 'nnue_minimax.dart';
import 'nnue_persistence_ref.dart';
// import 'chess_with_nnue.dart';
// import 'nnue_engine.dart';  // The engine class we created earlier

Future<void> main() async {
  const modelPath = "nnue_static_model.json";

  // Create NNUE-enabled engine
  final engine = NNUEEngine();

  // Load NNUE weights
  final loaded = await NNUESerializer.load(engine.nnue, modelPath);
  print(
    loaded
        ? "Loaded NNUE model from $modelPath"
        : "No model found. Using random NNUE weights.",
  );

  print("Initial NNUE eval: ${engine.eval()}");

  print("\n=== NNUE ENGINE READY ===");
  print(
    "Type a move in LAN (e.g. e2e4). Type 'go' for engine move. 'quit' to exit.\n",
  );

  while (true) {
    stdout.write(
      "${engine.turn == Color.WHITE ? 'White' : 'Black'} to move > ",
    );
    final input = stdin.readLineSync();

    if (input == null) continue;
    if (input == "quit") break;

    if (input == "go") {
      final mv = engine.play(depth: 2);
      print("Engine plays: $mv");
      continue;
    }

    // Human move
    final ok = engine.moveLAN(input);
    if (!ok) {
      print("Illegal move.");
      continue;
    }

    print("NNUE eval: ${engine.eval()}");
  }
}
