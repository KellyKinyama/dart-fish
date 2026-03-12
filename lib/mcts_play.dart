import 'dart:io';
import 'mcts.dart';

final Map<String, List<String>> commands = {
  "uci": ["id name Dartfish", "id author Hari Ambethkar", "uciok"],
  "isready": ["readyok"],
};

void run() {
  Engine bot = Engine();

  while (true) {
    String? cmd = stdin.readLineSync();

    if (cmd == null) {
      break;
    }

    cmd = cmd.trim();

    // Basic commands
    if (commands.containsKey(cmd)) {
      for (String out in commands[cmd]!) {
        print(out);
      }
      continue;
    }

    if (cmd == "ucinewgame") {
      bot = Engine();
      continue;
    }

    // GO command
    if (cmd.startsWith("go")) {
      String move = bot.play();

      if (move.isEmpty) {
        print("bestmove 0000");
      } else {
        print("bestmove $move");
      }

      continue;
    }

    // POSITION command
    if (cmd.startsWith("position")) {
      List<String> parts = cmd.split(" ");

      if (parts[1] == "startpos") {
        bot = Engine();

        int movesIndex = parts.indexOf("moves");
        if (movesIndex != -1) {
          for (int i = movesIndex + 1; i < parts.length; i++) {
            bot.moveLAN(parts[i]);
          }
        }
      }

      if (parts[1] == "fen") {
        int fenEnd = parts.indexOf("moves");

        String fen;

        if (fenEnd == -1) {
          fen = parts.sublist(2).join(" ");
        } else {
          fen = parts.sublist(2, fenEnd).join(" ");
        }

        bot = Engine.fromFEN(fen);

        if (fenEnd != -1) {
          for (int i = fenEnd + 1; i < parts.length; i++) {
            bot.moveLAN(parts[i]);
          }
        }
      }

      continue;
    }

    if (cmd == "quit") {
      break;
    }
  }
}

void main() {
  run();
}
