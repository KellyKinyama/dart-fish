import 'package:dart_fish/chess_nnue3.dart';
import 'package:dart_fish/mcts_nnue.dart';
import 'package:dart_fish/nnue_persistence2.dart';

Future<void> main() async {
  final game = ChessWithNNUE();

  // Optional: load weights if you have a serializer
  // await NNUESerializer.load(game, 'chess_model_v1.json');

  final mcts = MCTS(game, iterations: 4000, cPuct: 1.2);

  const String modelPath = 'chess_model_v1.json';

  // 1) Load weights if available
  try {
    await NNUESerializer.load(game.nnue, modelPath);
    print("Loaded NNUE model from $modelPath");
  } catch (e) {
    print("No existing model to load (or load failed): $e");
  }

  // Best move as LAN
  final bestLan = mcts.search();
  if (bestLan != null) {
    print('MCTS suggests: $bestLan');
    game.moveLAN(bestLan); // implement or reuse your own LAN applier
  } else {
    print('No legal moves found (game over).');
  }
}
