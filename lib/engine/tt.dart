import 'chess.dart';

enum Bound { EXACT, UPPER, LOWER }

class HashEntry {
  Move? move;
  double score;
  int hash;
  int depth;
  Bound bound;

  HashEntry({
    this.move,
    required this.hash,
    required this.score,
    this.depth = 0,
    required this.bound,
  });
}

class TTEntry {
  HashEntry? entry;

  void store(HashEntry newEntry) {
    if (entry == null) {
      entry = newEntry;
      return;
    }

    final old = entry!;

    // Deep → New rule
    if (newEntry.depth > old.depth) {
      entry = newEntry;
      return;
    }

    // Same depth: prefer exact bounds
    if (newEntry.depth == old.depth) {
      if (newEntry.bound == Bound.EXACT && old.bound != Bound.EXACT) {
        entry = newEntry;
        return;
      }

      // If still tied: NEW wins
      entry = newEntry;
    }
  }
}

class TranspositionTable {
  final int size;
  late List<TTEntry> table;

  TranspositionTable({this.size = 1 << 20}) {
    // ~1M entries
    table = List.generate(size, (_) => TTEntry());
  }

  TTEntry operator [](int hash) => table[hash & (size - 1)];

  void store(HashEntry entry) {
    table[entry.hash & (size - 1)].store(entry);
  }

  HashEntry? probe(int hash) {
    return table[hash & (size - 1)].entry;
  }
}
