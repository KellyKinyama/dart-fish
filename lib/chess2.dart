library chess;

import 'constants.dart';

import 'dart:math';

/*  Copyright (c) 2014, David Kopec (my first name at oaksnow dot com)
 *  Released under the MIT license
 *  https://github.com/davecom/chess.dart/blob/master/LICENSE
 *
 *  Based on chess.js
 *  Copyright (c) 2013, Jeff Hlywa (jhlywa@gmail.com)
 *  Released under the BSD license
 *  https://github.com/jhlywa/chess.js/blob/master/LICENSE
 */

class Chess {
  // Constants/Class Variables
  static const Color BLACK = Color.BLACK;
  static const Color WHITE = Color.WHITE;

  static const int EMPTY = -1;

  static const PieceType PAWN = PieceType.PAWN;
  static const PieceType KNIGHT = PieceType.KNIGHT;
  static const PieceType BISHOP = PieceType.BISHOP;
  static const PieceType ROOK = PieceType.ROOK;
  static const PieceType QUEEN = PieceType.QUEEN;
  static const PieceType KING = PieceType.KING;

  static const Map<String, PieceType> PIECE_TYPES = {
    'p': PieceType.PAWN,
    'n': PieceType.KNIGHT,
    'b': PieceType.BISHOP,
    'r': PieceType.ROOK,
    'q': PieceType.QUEEN,
    'k': PieceType.KING,
  };

  static const String SYMBOLS = 'pnbrqkPNBRQK';

  static const String DEFAULT_POSITION =
      'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

  static const List POSSIBLE_RESULTS = ['1-0', '0-1', '1/2-1/2', '*'];

  static const Map<Color, List<int>> PAWN_OFFSETS = {
    BLACK: [16, 32, 17, 15],
    WHITE: [-16, -32, -17, -15],
  };

  static const Map<PieceType, List<int>> PIECE_OFFSETS = {
    KNIGHT: [-18, -33, -31, -14, 18, 33, 31, 14],
    BISHOP: [-17, -15, 17, 15],
    ROOK: [-16, 1, 16, -1],
    QUEEN: [-17, -16, -15, 1, 17, 16, 15, -1],
    KING: [-17, -16, -15, 1, 17, 16, 15, -1],
  };

  static const List ATTACKS = [
    20,
    0,
    0,
    0,
    0,
    0,
    0,
    24,
    0,
    0,
    0,
    0,
    0,
    0,
    20,
    0,
    0,
    20,
    0,
    0,
    0,
    0,
    0,
    24,
    0,
    0,
    0,
    0,
    0,
    20,
    0,
    0,
    0,
    0,
    20,
    0,
    0,
    0,
    0,
    24,
    0,
    0,
    0,
    0,
    20,
    0,
    0,
    0,
    0,
    0,
    0,
    20,
    0,
    0,
    0,
    24,
    0,
    0,
    0,
    20,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    20,
    0,
    0,
    24,
    0,
    0,
    20,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    20,
    2,
    24,
    2,
    20,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    2,
    53,
    56,
    53,
    2,
    0,
    0,
    0,
    0,
    0,
    0,
    24,
    24,
    24,
    24,
    24,
    24,
    56,
    0,
    56,
    24,
    24,
    24,
    24,
    24,
    24,
    0,
    0,
    0,
    0,
    0,
    0,
    2,
    53,
    56,
    53,
    2,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    20,
    2,
    24,
    2,
    20,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    20,
    0,
    0,
    24,
    0,
    0,
    20,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    20,
    0,
    0,
    0,
    24,
    0,
    0,
    0,
    20,
    0,
    0,
    0,
    0,
    0,
    0,
    20,
    0,
    0,
    0,
    0,
    24,
    0,
    0,
    0,
    0,
    20,
    0,
    0,
    0,
    0,
    20,
    0,
    0,
    0,
    0,
    0,
    24,
    0,
    0,
    0,
    0,
    0,
    20,
    0,
    0,
    20,
    0,
    0,
    0,
    0,
    0,
    0,
    24,
    0,
    0,
    0,
    0,
    0,
    0,
    20,
  ];

  static final Map<Color, List> ROOKS = {
    WHITE: [
      {'square': SQUARES_A1, 'flag': BITS_QSIDE_CASTLE},
      {'square': SQUARES_H1, 'flag': BITS_KSIDE_CASTLE},
    ],
    BLACK: [
      {'square': SQUARES_A8, 'flag': BITS_QSIDE_CASTLE},
      {'square': SQUARES_H8, 'flag': BITS_KSIDE_CASTLE},
    ],
  };

  // Instance Variables
  List<Piece?> board = []..length = 128;
  ColorMap<int> kings = ColorMap(EMPTY);
  Color turn = WHITE;
  ColorMap<int> castling = ColorMap(0);
  int? ep_square = EMPTY;
  int half_moves = 0;
  int move_number = 1;
  List<State> history = [];
  Map header = {};

  /// By default start with the standard chess starting position
  Chess() {
    load(DEFAULT_POSITION);
  }

  /// Start with a position from a FEN
  Chess.fromFEN(String fen, {bool check_validity = true}) {
    load(fen, check_validity: check_validity);
  }

  /// Deep copy of the current Chess instance
  Chess copy() {
    return Chess()
      ..board = List<Piece?>.from(board)
      ..kings = ColorMap<int>.clone(kings)
      ..turn = turn
      ..castling = ColorMap<int>.clone(castling)
      ..ep_square = ep_square
      ..half_moves = half_moves
      ..move_number = move_number
      ..history = List<State>.from(history)
      ..header = Map.from(header);
  }

  /// Reset all of the instance variables
  void clear() {
    board = []..length = 128;
    kings = ColorMap(EMPTY);
    turn = WHITE;
    castling = ColorMap(0);
    ep_square = EMPTY;
    half_moves = 0;
    move_number = 1;
    history = [];
    header = {};
    update_setup(generate_fen());
  }

  /// Go back to the chess starting position
  void reset() {
    load(DEFAULT_POSITION);
  }

  /// Load a position from a FEN String
  bool load(String fen, {bool check_validity = true}) {
    List tokens = fen.split(RegExp(r'\s+'));
    String position = tokens[0];
    var square = 0;
    //String valid = SYMBOLS + '12345678/';

    if (check_validity) {
      final validMap = validate_fen(fen);
      if (!validMap['valid']) {
        print(validMap['error']);
        return false;
      }
    }

    clear();

    for (var i = 0; i < position.length; i++) {
      final piece = position[i];

      if (piece == '/') {
        square += 8;
      } else if (is_digit(piece)) {
        square += int.parse(piece);
      } else {
        var color = (piece == piece.toUpperCase()) ? WHITE : BLACK;
        var type = PIECE_TYPES[piece.toLowerCase()]!;
        put(Piece(type, color), algebraic(square));
        square++;
      }
    }

    if (tokens[1] == 'w') {
      turn = WHITE;
    } else {
      assert(tokens[1] == 'b');
      turn = BLACK;
    }

    if (tokens[2].indexOf('K') > -1) {
      castling[WHITE] |= BITS_KSIDE_CASTLE;
    }
    if (tokens[2].indexOf('Q') > -1) {
      castling[WHITE] |= BITS_QSIDE_CASTLE;
    }
    if (tokens[2].indexOf('k') > -1) {
      castling[BLACK] |= BITS_KSIDE_CASTLE;
    }
    if (tokens[2].indexOf('q') > -1) {
      castling[BLACK] |= BITS_QSIDE_CASTLE;
    }

    ep_square = (tokens[3] == '-') ? EMPTY : SQUARES[tokens[3]];
    half_moves = int.parse(tokens[4]);
    move_number = int.parse(tokens[5]);

    update_setup(generate_fen());

    zobristKey = generateZobristKey();

    return true;
  }

  /// Check the formatting of a FEN String is correct
  /// Returns a Map with keys valid, error_number, and error
  static Map validate_fen(String fen) {
    const errors = {
      0: 'No errors.',
      1: 'FEN string must contain six space-delimited fields.',
      2: '6th field (move number) must be a positive integer.',
      3: '5th field (half move counter) must be a non-negative integer.',
      4: '4th field (en-passant square) is invalid.',
      5: '3rd field (castling availability) is invalid.',
      6: '2nd field (side to move) is invalid.',
      7: '1st field (piece positions) does not contain 8 \'/\'-delimited rows.',
      8: '1st field (piece positions) is invalid [consecutive numbers].',
      9: '1st field (piece positions) is invalid [invalid piece].',
      10: '1st field (piece positions) is invalid [row too large].',
      11: '1st field (piece positions) is invalid [wrong kings counts]',
      12: '1st field (piece positions) is invalid [kings on neighbours cells]',
      13: '1st field (piece positions) is invalid [pawn(s) on first/last rank]',
      14: 'King of opponent player is in check.',
    };

    /* 1st criterion: 6 space-seperated fields? */
    List tokens = fen.split(RegExp(r'\s+'));
    if (tokens.length != 6) {
      return {'valid': false, 'error_number': 1, 'error': errors[1]};
    }

    /* 2nd criterion: move number field is a integer value > 0? */
    var temp = int.tryParse(tokens[5]);
    if (temp != null) {
      if (temp <= 0) {
        return {'valid': false, 'error_number': 2, 'error': errors[2]};
      }
    } else {
      return {'valid': false, 'error_number': 2, 'error': errors[2]};
    }

    /* 3rd criterion: half move counter is an integer >= 0? */
    temp = int.tryParse(tokens[4]);
    if (temp != null) {
      if (temp < 0) {
        return {'valid': false, 'error_number': 3, 'error': errors[3]};
      }
    } else {
      return {'valid': false, 'error_number': 3, 'error': errors[3]};
    }

    /* 4th criterion: 4th field is a valid e.p.-string? */
    final check4 = RegExp(r'^(-|[abcdefgh][36])$');
    if (check4.firstMatch(tokens[3]) == null) {
      return {'valid': false, 'error_number': 4, 'error': errors[4]};
    }

    /* 5th criterion: 3th field is a valid castle-string? */
    final check5 = RegExp(r'^(KQ?k?q?|Qk?q?|kq?|q|-)$');
    if (check5.firstMatch(tokens[2]) == null) {
      return {'valid': false, 'error_number': 5, 'error': errors[5]};
    }

    /* 6th criterion: 2nd field is "w" (white) or "b" (black)? */
    var check6 = RegExp(r'^([wb])$');
    if (check6.firstMatch(tokens[1]) == null) {
      return {'valid': false, 'error_number': 6, 'error': errors[6]};
    }

    /* 7th criterion: 1st field contains 8 rows? */
    List rows = tokens[0].split('/');
    if (rows.length != 8) {
      return {'valid': false, 'error_number': 7, 'error': errors[7]};
    }

    /* 8th criterion: every row is valid? */
    for (var i = 0; i < rows.length; i++) {
      /* check for right sum of fields AND not two numbers in succession */
      var sum_fields = 0;
      var previous_was_number = false;

      for (var k = 0; k < rows[i].length; k++) {
        final temp2 = int.tryParse(rows[i][k]);
        if (temp2 != null) {
          if (previous_was_number) {
            return {'valid': false, 'error_number': 8, 'error': errors[8]};
          }
          sum_fields += temp2;
          previous_was_number = true;
        } else {
          final checkOM = RegExp(r'^[prnbqkPRNBQK]$');
          if (checkOM.firstMatch(rows[i][k]) == null) {
            return {'valid': false, 'error_number': 9, 'error': errors[9]};
          }
          sum_fields += 1;
          previous_was_number = false;
        }
      }

      if (sum_fields != 8) {
        return {'valid': false, 'error_number': 10, 'error': errors[10]};
      }
    }

    final boardPart = fen.split(' ')[0];

    /* Is white and black kings' count legal (except for empty board) ? */
    final isEmptyBoard = boardPart == '8/8/8/8/8/8/8/8';
    final whiteKingCount = boardPart
        .split('')
        .where((elem) => elem == 'K')
        .length;
    final blackKingCount = boardPart
        .split('')
        .where((elem) => elem == 'k')
        .length;

    if (!isEmptyBoard && (whiteKingCount != 1 || blackKingCount != 1)) {
      return {'valid': false, 'error_number': 11, 'error': errors[11]};
    }

    /* Are both kings on neighbours cells ? */
    // Computes a kind of 'expanded' FEN : cells are translated as underscores,
    //  and removing all slashes.
    final expandedFen = boardPart.split('').fold<String>('', (accum, curr) {
      final digitValue = int.tryParse(curr);
      if (curr == '/') {
        return accum;
      } else if (digitValue != null) {
        var result = '';
        for (var i = 0; i < digitValue; i++) {
          result += '_';
        }
        return accum + result;
      } else {
        return accum + curr;
      }
    });
    final whiteKingIndex = expandedFen.indexOf('K');
    final blackKingIndex = expandedFen.indexOf('k');
    final whiteKingCoords = [whiteKingIndex % 8, whiteKingIndex ~/ 8];
    final blackKingCoords = [blackKingIndex % 8, blackKingIndex ~/ 8];

    final deltaX = (whiteKingCoords[0] - blackKingCoords[0]).abs();
    final deltaY = (whiteKingCoords[1] - blackKingCoords[1]).abs();

    final kingsTooClose = (deltaX <= 1) && (deltaY <= 1);
    if (!isEmptyBoard && kingsTooClose) {
      return {'valid': false, 'error_number': 12, 'error': errors[12]};
    }

    /* Any pawn on first or last rank ? */
    final firstRank = boardPart.split('/')[0];
    final lastRank = boardPart.split('/')[7];

    final whitePawnOnFirstRank = firstRank.contains('P');
    final blackPawnOnFirstRank = firstRank.contains('p');
    final whitePawnOnLastRank = lastRank.contains('P');
    final blackPawnOnLastRank = lastRank.contains('p');

    if (whitePawnOnFirstRank ||
        whitePawnOnLastRank ||
        blackPawnOnFirstRank ||
        blackPawnOnLastRank) {
      return {'valid': false, 'error_number': 13, 'error': errors[13]};
    }

    /* Is king of player in turn in check (without being in checkmate) ? */
    final board = Chess.fromFEN(fen, check_validity: false);
    final turn = board.turn;
    final opponentTurn = turn == Color.WHITE ? Color.BLACK : Color.WHITE;
    final kingInChess =
        !board.in_checkmate && board.king_attacked(opponentTurn);

    if (kingInChess) {
      return {'valid': false, 'error_number': 14, 'error': errors[14]};
    }

    /* everything's okay! */
    return {'valid': true, 'error_number': 0, 'error': errors[0]};
  }

  /// Returns a FEN String representing the current position
  String generate_fen() {
    var empty = 0;
    var fen = '';

    for (var i = SQUARES_A8; i <= SQUARES_H1; i++) {
      if (board[i] == null) {
        empty++;
      } else {
        if (empty > 0) {
          fen += empty.toString();
          empty = 0;
        }
        var color = board[i]!.color;
        PieceType? type = board[i]!.type;

        fen += (color == WHITE) ? type.toUpperCase() : type.toLowerCase();
      }

      if (((i + 1) & 0x88) != 0) {
        if (empty > 0) {
          fen += empty.toString();
        }

        if (i != SQUARES_H1) {
          fen += '/';
        }

        empty = 0;
        i += 8;
      }
    }

    var cflags = '';
    if ((castling[WHITE] & BITS_KSIDE_CASTLE) != 0) {
      cflags += 'K';
    }
    if ((castling[WHITE] & BITS_QSIDE_CASTLE) != 0) {
      cflags += 'Q';
    }
    if ((castling[BLACK] & BITS_KSIDE_CASTLE) != 0) {
      cflags += 'k';
    }
    if ((castling[BLACK] & BITS_QSIDE_CASTLE) != 0) {
      cflags += 'q';
    }

    /* do we have an empty castling flag? */
    if (cflags == '') {
      cflags = '-';
    }
    final epflags = (ep_square == EMPTY) ? '-' : algebraic(ep_square!);
    final turnStr = (turn == Color.WHITE) ? 'w' : 'b';

    return [fen, turnStr, cflags, epflags, half_moves, move_number].join(' ');
  }

  /// Updates [header] with the List of args and returns it
  Map set_header(args) {
    for (var i = 0; i < args.length; i += 2) {
      if (args[i] is String && args[i + 1] is String) {
        header[args[i]] = args[i + 1];
      }
    }
    return header;
  }

  /// called when the initial board setup is changed with put() or remove().
  /// modifies the SetUp and FEN properties of the header object.  if the FEN is
  /// equal to the default position, the SetUp and FEN are deleted
  /// the setup is only updated if history.length is zero, ie moves haven't been
  /// made.
  void update_setup(String fen) {
    if (history.isNotEmpty) return;

    if (fen != DEFAULT_POSITION) {
      header['SetUp'] = '1';
      header['FEN'] = fen;
    } else {
      header.remove('SetUp');
      header.remove('FEN');
    }
  }

  /// Returns the piece at the square in question or null
  /// if there is none
  Piece? get(String square) {
    return board[SQUARES[square]];
  }

  /// Put [piece] on [square]
  bool put(Piece piece, String square) {
    /* check for piece */
    if (!SYMBOLS.contains(piece.type.toLowerCase())) {
      return false;
    }

    /* check for valid square */
    if (!(SQUARES.containsKey(square))) {
      return false;
    }

    int sq = SQUARES[square];
    board[sq] = piece;
    if (piece.type == KING) {
      kings[piece.color] = sq;
    }

    update_setup(generate_fen());

    return true;
  }

  /// Removes a piece from a square and returns it,
  /// or null if none is present
  Piece? remove(String square) {
    final piece = get(square);
    board[SQUARES[square]] = null;
    if (piece != null && piece.type == KING) {
      kings[piece.color] = EMPTY;
    }

    update_setup(generate_fen());

    return piece;
  }

  Move build_move(List<Piece?> board, from, to, flags, [PieceType? promotion]) {
    if (promotion != null) {
      flags |= BITS_PROMOTION;
    }

    PieceType? captured;
    final toPiece = board[to];
    if (toPiece != null) {
      captured = toPiece.type;
    } else if ((flags & BITS_EP_CAPTURE) != 0) {
      captured = PAWN;
    }
    return Move(turn, from, to, flags, board[from]!.type, captured, promotion);
  }

  List<Move> generate_moves([Map? options]) {
    void add_move(List<Piece?> board, List<Move> moves, from, to, flags) {
      /* if pawn promotion */
      if (board[from]!.type == PAWN &&
          (rank(to) == RANK_8 || rank(to) == RANK_1)) {
        const pieces = [QUEEN, ROOK, BISHOP, KNIGHT];
        for (var i = 0, len = pieces.length; i < len; i++) {
          moves.add(build_move(board, from, to, flags, pieces[i]));
        }
      } else {
        moves.add(build_move(board, from, to, flags));
      }
    }

    final moves = <Move>[];
    final us = turn;
    final them = swap_color(us);
    final second_rank = ColorMap<int>(0);
    second_rank[BLACK] = RANK_7;
    second_rank[WHITE] = RANK_2;

    var first_sq = SQUARES_A8;
    var last_sq = SQUARES_H1;
    var single_square = false;

    /* do we want legal moves? */
    final legal = (options != null && options.containsKey('legal'))
        ? options['legal']
        : true;

    /* are we generating moves for a single square? */
    if (options != null && options.containsKey('square')) {
      if (SQUARES.containsKey(options['square'])) {
        first_sq = last_sq = SQUARES[options['square']];
        single_square = true;
      } else {
        /* invalid square */
        return [];
      }
    }

    for (var i = first_sq; i <= last_sq; i++) {
      /* did we run off the end of the board */
      if ((i & 0x88) != 0) {
        i += 7;
        continue;
      }

      final piece = board[i];
      if (piece == null || piece.color != us) {
        continue;
      }

      if (piece.type == PAWN) {
        /* single square, non-capturing */
        final square = i + PAWN_OFFSETS[us]![0];
        if (board[square] == null) {
          add_move(board, moves, i, square, BITS_NORMAL);

          /* double square */
          final square2 = i + PAWN_OFFSETS[us]![1];
          if (second_rank[us] == rank(i) && board[square2] == null) {
            add_move(board, moves, i, square2, BITS_BIG_PAWN);
          }
        }

        /* pawn captures */
        for (var j = 2; j < 4; j++) {
          var square = i + PAWN_OFFSETS[us]![j];
          if ((square & 0x88) != 0) continue;

          if (board[square] != null && board[square]!.color == them) {
            add_move(board, moves, i, square, BITS_CAPTURE);
          } else if (square == ep_square) {
            add_move(board, moves, i, ep_square, BITS_EP_CAPTURE);
          }
        }
      } else {
        for (var j = 0, len = PIECE_OFFSETS[piece.type]!.length; j < len; j++) {
          final offset = PIECE_OFFSETS[piece.type]![j];
          var square = i;

          while (true) {
            square += offset;
            if ((square & 0x88) != 0) break;

            if (board[square] == null) {
              add_move(board, moves, i, square, BITS_NORMAL);
            } else {
              if (board[square]!.color == us) {
                break;
              }
              add_move(board, moves, i, square, BITS_CAPTURE);
              break;
            }

            /* break, if knight or king */
            if (piece.type == KNIGHT || piece.type == KING) break;
          }
        }
      }
    }

    // check for castling if: a) we're generating all moves, or b) we're doing
    // single square move generation on the king's square
    if ((!single_square) || last_sq == kings[us]) {
      /* king-side castling */
      if ((castling[us] & BITS_KSIDE_CASTLE) != 0) {
        final castling_from = kings[us];
        final castling_to = castling_from + 2;

        if (board[castling_from + 1] == null &&
            board[castling_to] == null &&
            !attacked(them, kings[us]) &&
            !attacked(them, castling_from + 1) &&
            !attacked(them, castling_to)) {
          add_move(board, moves, kings[us], castling_to, BITS_KSIDE_CASTLE);
        }
      }

      /* queen-side castling */
      if ((castling[us] & BITS_QSIDE_CASTLE) != 0) {
        final castling_from = kings[us];
        final castling_to = castling_from - 2;

        if (board[castling_from - 1] == null &&
            board[castling_from - 2] == null &&
            board[castling_from - 3] == null &&
            !attacked(them, kings[us]) &&
            !attacked(them, castling_from - 1) &&
            !attacked(them, castling_to)) {
          add_move(board, moves, kings[us], castling_to, BITS_QSIDE_CASTLE);
        }
      }
    }

    /* return all pseudo-legal moves (this includes moves that allow the king
     * to be captured)
     */
    if (!legal) {
      return moves;
    }

    /* filter out illegal moves */
    final legal_moves = <Move>[];
    for (var i = 0, len = moves.length; i < len; i++) {
      make_move(moves[i]);
      if (!king_attacked(us)) {
        legal_moves.add(moves[i]);
      }
      undo_move();
    }

    return legal_moves;
  }

  /// Convert a move from 0x88 coordinates to Standard Algebraic Notation(SAN)
  String move_to_san(Move move) {
    var output = '';
    final flags = move.flags;
    if ((flags & BITS_KSIDE_CASTLE) != 0) {
      output = 'O-O';
    } else if ((flags & BITS_QSIDE_CASTLE) != 0) {
      output = 'O-O-O';
    } else {
      var disambiguator = get_disambiguator(move);

      if (move.piece != PAWN) {
        output += move.piece.toUpperCase() + disambiguator;
      }

      if ((flags & (BITS_CAPTURE | BITS_EP_CAPTURE)) != 0) {
        if (move.piece == PAWN) {
          output += move.fromAlgebraic[0];
        }
        output += 'x';
      }

      output += move.toAlgebraic;

      if ((flags & BITS_PROMOTION) != 0) {
        output += '=' + move.promotion!.toUpperCase();
      }
    }

    make_move(move);
    if (in_check) {
      if (in_checkmate) {
        output += '#';
      } else {
        output += '+';
      }
    }
    undo_move();

    return output;
  }

  bool attacked(Color color, int square) {
    for (var i = SQUARES_A8; i <= SQUARES_H1; i++) {
      /* did we run off the end of the board */
      if ((i & 0x88) != 0) {
        i += 7;
        continue;
      }

      /* if empty square or wrong color */
      final piece = board[i];
      if (piece == null || piece.color != color) continue;

      final difference = i - square;
      final index = difference + 119;
      final type = piece.type;

      if ((ATTACKS[index] & (1 << type.shift)) != 0) {
        if (type == PAWN) {
          if (difference > 0) {
            if (color == WHITE) return true;
          } else {
            if (color == BLACK) return true;
          }
          continue;
        }

        /* if the piece is a knight or a king */
        if (type == KNIGHT || type == KING) return true;

        final offset = RAYS[index];
        var j = i + offset;

        var blocked = false;
        while (j != square) {
          if (board[j] != null) {
            blocked = true;
            break;
          }
          j += offset;
        }

        if (!blocked) return true;
      }
    }

    return false;
  }

  bool king_attacked(Color color) {
    return attacked(swap_color(color), kings[color]);
  }

  bool get in_check {
    return king_attacked(turn);
  }

  bool get in_checkmate {
    return in_check && generate_moves().isEmpty;
  }

  bool get in_stalemate {
    return !in_check && generate_moves().isEmpty;
  }

  bool get insufficient_material {
    final pieces = {};
    final bishops = <int>[];
    var num_pieces = 0;
    var sq_color = 0;

    for (var i = SQUARES_A8; i <= SQUARES_H1; i++) {
      sq_color = (sq_color + 1) % 2;
      if ((i & 0x88) != 0) {
        i += 7;
        continue;
      }

      var piece = board[i];
      if (piece != null) {
        pieces[piece.type] = (pieces.containsKey(piece.type))
            ? pieces[piece.type] + 1
            : 1;
        if (piece.type == BISHOP) {
          bishops.add(sq_color);
        }
        num_pieces++;
      }
    }

    /* k vs. k */
    if (num_pieces == 2) {
      return true;
    } /* k vs. kn .... or .... k vs. kb */ else if (num_pieces == 3 &&
        (pieces[BISHOP] == 1 || pieces[KNIGHT] == 1)) {
      return true;
    } /* kb vs. kb where any number of bishops are all on the same color */ else if (pieces
            .containsKey(BISHOP) &&
        num_pieces == (pieces[BISHOP] + 2)) {
      var sum = 0;
      var len = bishops.length;
      for (var i = 0; i < len; i++) {
        sum += bishops[i];
      }
      if (sum == 0 || sum == len) {
        return true;
      }
    }

    return false;
  }

  bool get in_threefold_repetition {
    /* TODO: while this function is fine for casual use, a better
     * implementation would use a Zobrist key (instead of FEN). the
     * Zobrist key would be maintained in the make_move/undo_move functions,
     * avoiding the costly that we do below.
     */
    final positions = {};
    var moves = [];
    var repetition = false;

    while (true) {
      var move = undo_move();
      if (move == null) {
        break;
      }
      moves.add(move);
    }

    while (true) {
      /* remove the last two fields in the FEN string, they're not needed
       * when checking for draw by rep */
      var fen = generate_fen().split(' ').sublist(0, 4).join(' ');

      /* has the position occurred three or move times */
      positions[fen] = (positions.containsKey(fen)) ? positions[fen] + 1 : 1;
      if (positions[fen] >= 3) {
        repetition = true;
      }

      if (moves.isEmpty) {
        break;
      }
      make_move(moves.removeLast());
    }

    return repetition;
  }

  void push(Move move) {
    history.add(
      State(
        move,
        ColorMap.clone(kings),
        turn,
        ColorMap.clone(castling),
        ep_square,
        half_moves,
        move_number,
        zobristKey, // Save current hash to history
      ),
    );
  }

  // void make_move(Move move) {
  //   final us = turn;
  //   final them = swap_color(us);
  //   push(move);

  //   board[move.to] = board[move.from];
  //   board[move.from] = null;

  //   /* if ep capture, remove the captured pawn */
  //   if ((move.flags & BITS_EP_CAPTURE) != 0) {
  //     if (turn == BLACK) {
  //       board[move.to - 16] = null;
  //     } else {
  //       board[move.to + 16] = null;
  //     }
  //   }

  //   /* if pawn promotion, replace with new piece */
  //   if ((move.flags & BITS_PROMOTION) != 0) {
  //     board[move.to] = Piece(move.promotion!, us);
  //   }

  //   /* if we moved the king */
  //   if (board[move.to]!.type == KING) {
  //     kings[board[move.to]!.color] = move.to;

  //     /* if we castled, move the rook next to the king */
  //     if ((move.flags & BITS_KSIDE_CASTLE) != 0) {
  //       final castling_to = move.to - 1;
  //       final castling_from = move.to + 1;
  //       board[castling_to] = board[castling_from];
  //       board[castling_from] = null;
  //     } else if ((move.flags & BITS_QSIDE_CASTLE) != 0) {
  //       final castling_to = move.to + 1;
  //       final castling_from = move.to - 2;
  //       board[castling_to] = board[castling_from];
  //       board[castling_from] = null;
  //     }

  //     /* turn off castling */
  //     castling[us] = 0;
  //   }

  //   /* turn off castling if we move a rook */
  //   if (castling[us] != 0) {
  //     for (var i = 0, len = ROOKS[us]!.length; i < len; i++) {
  //       if (move.from == ROOKS[us]![i]['square'] &&
  //           ((castling[us] & ROOKS[us]![i]['flag']) != 0)) {
  //         castling[us] ^= ROOKS[us]![i]['flag'];
  //         break;
  //       }
  //     }
  //   }

  //   /* turn off castling if we capture a rook */
  //   if (castling[them] != 0) {
  //     for (var i = 0, len = ROOKS[them]!.length; i < len; i++) {
  //       if (move.to == ROOKS[them]![i]['square'] &&
  //           ((castling[them] & ROOKS[them]![i]['flag']) != 0)) {
  //         castling[them] ^= ROOKS[them]![i]['flag'];
  //         break;
  //       }
  //     }
  //   }

  //   /* if big pawn move, update the en passant square */
  //   if ((move.flags & BITS_BIG_PAWN) != 0) {
  //     if (turn == BLACK) {
  //       ep_square = move.to - 16;
  //     } else {
  //       ep_square = move.to + 16;
  //     }
  //   } else {
  //     ep_square = EMPTY;
  //   }

  //   /* reset the 50 move counter if a pawn is moved or a piece is captured */
  //   if (move.piece == PAWN) {
  //     half_moves = 0;
  //   } else if ((move.flags & (BITS_CAPTURE | BITS_EP_CAPTURE)) != 0) {
  //     half_moves = 0;
  //   } else {
  //     half_moves++;
  //   }

  //   if (turn == BLACK) {
  //     move_number++;
  //   }
  //   turn = swap_color(turn);
  // }

  // void make_move(Move move) {
  //   final us = turn;
  //   final them = swap_color(us);

  //   // --- 1. CAPTURE STATE BEFORE MODIFICATIONS ---
  //   // We push to history FIRST so that old.zobristKey stores the pure
  //   // hash of the board before this move started.
  //   push(move);

  //   // --- 2. XOR OUT OLD METADATA ---
  //   // Turn
  //   if (turn == BLACK) zobristKey ^= _zobristTurn;

  //   // Castling (using the & 3 mask to prevent RangeErrors)
  //   zobristKey ^=
  //       _zobristCastling[((castling[WHITE] & 3) << 2) | (castling[BLACK] & 3)];

  //   // En Passant
  //   if (ep_square != EMPTY && ep_square != null) {
  //     zobristKey ^= _zobristEnPassant[ep_square!];
  //   }

  //   // --- 3. PIECE UPDATES ---
  //   // XOR out the moving piece from its origin
  //   zobristKey ^= _getPieceHash(board[move.from]!, move.from);

  //   // If this is a capture, XOR out the piece being replaced
  //   if ((move.flags & BITS_CAPTURE) != 0) {
  //     zobristKey ^= _getPieceHash(board[move.to]!, move.to);
  //   }

  //   // Handle En Passant capture (XOR out the pawn on the adjacent rank)
  //   if ((move.flags & BITS_EP_CAPTURE) != 0) {
  //     final capSq = (us == WHITE) ? move.to + 16 : move.to - 16;
  //     zobristKey ^= _getPieceHash(Piece(PAWN, them), capSq);
  //     board[capSq] = null;
  //   }

  //   // Move the piece on the board
  //   board[move.to] = board[move.from];
  //   board[move.from] = null;

  //   // Handle Promotion
  //   if ((move.flags & BITS_PROMOTION) != 0) {
  //     // The pawn was moved to move.to; XOR it out and replace with promoted piece
  //     // (Note: Since we already moved board[move.from] to board[move.to],
  //     // board[move.to] currently holds a pawn)
  //     zobristKey ^= _getPieceHash(board[move.to]!, move.to); // Remove Pawn
  //     board[move.to] = Piece(move.promotion!, us);
  //     zobristKey ^= _getPieceHash(
  //       board[move.to]!,
  //       move.to,
  //     ); // Add Promoted Piece
  //   } else {
  //     // Standard move: XOR the piece into its new destination
  //     zobristKey ^= _getPieceHash(board[move.to]!, move.to);
  //   }

  //   // --- 4. SPECIAL MOVES (Castling) ---
  //   if (board[move.to]!.type == KING) {
  //     kings[us] = move.to;

  //     if ((move.flags & BITS_KSIDE_CASTLE) != 0) {
  //       final cTo = move.to - 1, cFrom = move.to + 1;
  //       zobristKey ^= _getPieceHash(board[cFrom]!, cFrom);
  //       board[cTo] = board[cFrom];
  //       zobristKey ^= _getPieceHash(board[cTo]!, cTo);
  //       board[cFrom] = null;
  //     } else if ((move.flags & BITS_QSIDE_CASTLE) != 0) {
  //       final cTo = move.to + 1, cFrom = move.to - 2;
  //       zobristKey ^= _getPieceHash(board[cFrom]!, cFrom);
  //       board[cTo] = board[cFrom];
  //       zobristKey ^= _getPieceHash(board[cTo]!, cTo);
  //       board[cFrom] = null;
  //     }
  //     castling[us] = 0; // King move loses all castling rights
  //   }

  //   // --- 5. UPDATE CASTLING RIGHTS ---
  //   // If a Rook moved from its starting square
  //   if (castling[us] != 0) {
  //     for (var i = 0; i < ROOKS[us]!.length; i++) {
  //       if (move.from == ROOKS[us]![i]['square'] &&
  //           (castling[us] & ROOKS[us]![i]['flag']) != 0) {
  //         castling[us] ^= ROOKS[us]![i]['flag'];
  //         break;
  //       }
  //     }
  //   }

  //   // If a Rook was captured on its starting square
  //   if (castling[them] != 0) {
  //     for (var i = 0; i < ROOKS[them]!.length; i++) {
  //       if (move.to == ROOKS[them]![i]['square'] &&
  //           (castling[them] & ROOKS[them]![i]['flag']) != 0) {
  //         castling[them] ^= ROOKS[them]![i]['flag'];
  //         break;
  //       }
  //     }
  //   }

  //   // --- 6. UPDATE EP SQUARE ---
  //   if ((move.flags & BITS_BIG_PAWN) != 0) {
  //     ep_square = (us == WHITE) ? move.to + 16 : move.to - 16;
  //   } else {
  //     ep_square = EMPTY;
  //   }

  //   // --- 7. MISC STATE ---
  //   if (move.piece == PAWN ||
  //       (move.flags & (BITS_CAPTURE | BITS_EP_CAPTURE)) != 0) {
  //     half_moves = 0;
  //   } else {
  //     half_moves++;
  //   }

  //   if (us == BLACK) move_number++;
  //   turn = them;

  //   // --- 8. XOR IN NEW METADATA ---
  //   if (turn == BLACK) zobristKey ^= _zobristTurn;

  //   zobristKey ^=
  //       _zobristCastling[((castling[WHITE] & 3) << 2) | (castling[BLACK] & 3)];

  //   if (ep_square != EMPTY && ep_square != null) {
  //     zobristKey ^= _zobristEnPassant[ep_square!];
  //   }
  // }
  /// Helper to map the large bitflags (32, 64) into a clean 4-bit index (0-15)
  // int _getCastlingIndex() {
  //   int index = 0;
  //   if ((castling[WHITE] & BITS_KSIDE_CASTLE) != 0) index |= 1;
  //   if ((castling[WHITE] & BITS_QSIDE_CASTLE) != 0) index |= 2;
  //   if ((castling[BLACK] & BITS_KSIDE_CASTLE) != 0) index |= 4;
  //   if ((castling[BLACK] & BITS_QSIDE_CASTLE) != 0) index |= 8;
  //   return index;
  // }

  void make_move(Move move) {
    final us = turn;
    final them = swap_color(us);

    // --- 1. CAPTURE STATE BEFORE MODIFICATIONS ---
    push(move);

    // --- 2. XOR OUT OLD METADATA ---
    if (turn == BLACK) zobristKey ^= _zobristTurn;

    // Use the helper for path-independent castling hashes
    zobristKey ^= _zobristCastling[_getCastlingIndex()];

    if (ep_square != EMPTY && ep_square != null) {
      zobristKey ^= _zobristEnPassant[ep_square!];
    }

    // --- 3. PIECE UPDATES ---
    // XOR out the moving piece from its origin
    zobristKey ^= _getPieceHash(board[move.from]!, move.from);

    // If this is a capture, XOR out the piece being replaced
    if ((move.flags & BITS_CAPTURE) != 0) {
      zobristKey ^= _getPieceHash(board[move.to]!, move.to);
    }

    // Handle En Passant capture
    if ((move.flags & BITS_EP_CAPTURE) != 0) {
      final capSq = (us == WHITE) ? move.to + 16 : move.to - 16;
      zobristKey ^= _getPieceHash(Piece(PAWN, them), capSq);
      board[capSq] = null;
    }

    // Move the piece on the board
    board[move.to] = board[move.from];
    board[move.from] = null;

    // Handle Promotion
    if ((move.flags & BITS_PROMOTION) != 0) {
      zobristKey ^= _getPieceHash(board[move.to]!, move.to); // Remove Pawn
      board[move.to] = Piece(move.promotion!, us);
      zobristKey ^= _getPieceHash(
        board[move.to]!,
        move.to,
      ); // Add Promoted Piece
    } else {
      // Standard move: XOR the piece into its new destination
      zobristKey ^= _getPieceHash(board[move.to]!, move.to);
    }

    // --- 4. SPECIAL MOVES (Castling) ---
    if (board[move.to]!.type == KING) {
      kings[us] = move.to;

      if ((move.flags & BITS_KSIDE_CASTLE) != 0) {
        final cTo = move.to - 1, cFrom = move.to + 1;
        zobristKey ^= _getPieceHash(board[cFrom]!, cFrom);
        board[cTo] = board[cFrom];
        zobristKey ^= _getPieceHash(board[cTo]!, cTo);
        board[cFrom] = null;
      } else if ((move.flags & BITS_QSIDE_CASTLE) != 0) {
        final cTo = move.to + 1, cFrom = move.to - 2;
        zobristKey ^= _getPieceHash(board[cFrom]!, cFrom);
        board[cTo] = board[cFrom];
        zobristKey ^= _getPieceHash(board[cTo]!, cTo);
        board[cFrom] = null;
      }
      castling[us] = 0; // King move loses all rights
    }

    // --- 5. UPDATE CASTLING RIGHTS (Logic Check) ---
    if (castling[us] != 0) {
      for (var i = 0; i < ROOKS[us]!.length; i++) {
        if (move.from == ROOKS[us]![i]['square']) {
          // Use bitwise AND NOT to clear the specific flag
          castling[us] &= ~ROOKS[us]![i]['flag'];
          break;
        }
      }
    }

    if (castling[them] != 0) {
      for (var i = 0; i < ROOKS[them]!.length; i++) {
        if (move.to == ROOKS[them]![i]['square']) {
          castling[them] &= ~ROOKS[them]![i]['flag'];
          break;
        }
      }
    }

    // --- 6. UPDATE EP SQUARE ---
    if ((move.flags & BITS_BIG_PAWN) != 0) {
      ep_square = (us == WHITE) ? move.to + 16 : move.to - 16;
    } else {
      ep_square = EMPTY;
    }

    // --- 7. MISC STATE ---
    if (move.piece == PAWN ||
        (move.flags & (BITS_CAPTURE | BITS_EP_CAPTURE)) != 0) {
      half_moves = 0;
    } else {
      half_moves++;
    }

    if (us == BLACK) move_number++;
    turn = them;

    // --- 8. XOR IN NEW METADATA ---
    if (turn == BLACK) zobristKey ^= _zobristTurn;

    // XOR in the NEW mapped castling index
    zobristKey ^= _zobristCastling[_getCastlingIndex()];

    if (ep_square != EMPTY && ep_square != null) {
      zobristKey ^= _zobristEnPassant[ep_square!];
    }
  }

  /// Undoes a move and returns it, or null if move history is empty
  Move? undo_move() {
    if (history.isEmpty) return null;

    final old = history.removeLast();
    final move = old.move;

    kings = old.kings;
    turn = old.turn;
    castling = old.castling;
    ep_square = old.ep_square;
    half_moves = old.half_moves;
    move_number = old.move_number;
    zobristKey = old.zobristKey; // Restore the hash perfectly

    final us = turn;
    final them = swap_color(turn);

    board[move.from] = board[move.to];
    board[move.from]!.type = move.piece;
    board[move.to] = null;

    if ((move.flags & BITS_CAPTURE) != 0) {
      board[move.to] = Piece(move.captured!, them);
    } else if ((move.flags & BITS_EP_CAPTURE) != 0) {
      final index = (us == BLACK) ? move.to - 16 : move.to + 16;
      board[index] = Piece(PAWN, them);
    }

    if ((move.flags & (BITS_KSIDE_CASTLE | BITS_QSIDE_CASTLE)) != 0) {
      int castling_to, castling_from;
      if ((move.flags & BITS_KSIDE_CASTLE) != 0) {
        castling_to = move.to + 1;
        castling_from = move.to - 1;
      } else {
        castling_to = move.to - 2;
        castling_from = move.to + 1;
      }
      board[castling_to] = board[castling_from];
      board[castling_from] = null;
    }

    return move;
  }

  /* this function is used to uniquely identify ambiguous moves */
  String get_disambiguator(Move move) {
    var moves = generate_moves();

    var from = move.from;
    var to = move.to;
    var piece = move.piece;

    var ambiguities = 0;
    var same_rank = 0;
    var same_file = 0;

    for (var i = 0, len = moves.length; i < len; i++) {
      var ambig_from = moves[i].from;
      var ambig_to = moves[i].to;
      var ambig_piece = moves[i].piece;

      /* if a move of the same piece type ends on the same to square, we'll
       * need to add a disambiguator to the algebraic notation
       */
      if (piece == ambig_piece && from != ambig_from && to == ambig_to) {
        ambiguities++;

        if (rank(from) == rank(ambig_from)) {
          same_rank++;
        }

        if (file(from) == file(ambig_from)) {
          same_file++;
        }
      }
    }

    if (ambiguities > 0) {
      /* if there exists a similar moving piece on the same rank and file as
       * the move in question, use the square as the disambiguator
       */
      if (same_rank > 0 && same_file > 0) {
        return algebraic(from);
      } /* if the moving piece rests on the same file, use the rank symbol as the
       * disambiguator
       */ else if (same_file > 0) {
        return algebraic(from)[1];
      } /* else use the file symbol */ else {
        return algebraic(from)[0];
      }
    }

    return '';
  }

  /// Returns a String representation of the current position
  /// complete with ascii art
  String get ascii {
    var s = '   +------------------------+\n';
    for (var i = SQUARES_A8; i <= SQUARES_H1; i++) {
      /* display the rank */
      if (file(i) == 0) {
        s += ' ' + '87654321'[rank(i)] + ' |';
      }

      /* empty piece */
      if (board[i] == null) {
        s += ' . ';
      } else {
        var type = board[i]!.type;
        var color = board[i]!.color;
        var symbol = (color == WHITE) ? type.toUpperCase() : type.toLowerCase();
        s += ' ' + symbol + ' ';
      }

      if (((i + 1) & 0x88) != 0) {
        s += '|\n';
        i += 8;
      }
    }
    s += '   +------------------------+\n';
    s += '     a  b  c  d  e  f  g  h\n';

    return s;
  }

  // Utility Functions
  static int rank(int i) {
    return i >> 4;
  }

  static int file(int i) {
    return i & 15;
  }

  static String algebraic(int i) {
    var f = file(i), r = rank(i);
    return 'abcdefgh'.substring(f, f + 1) + '87654321'.substring(r, r + 1);
  }

  static Color swap_color(Color c) {
    return c == WHITE ? BLACK : WHITE;
  }

  static bool is_digit(String c) {
    return '0123456789'.contains(c);
  }

  /// pretty = external move object
  Map<String, dynamic> make_pretty(Move ugly_move) {
    final map = <String, dynamic>{};
    map['san'] = move_to_san(ugly_move);
    map['to'] = ugly_move.toAlgebraic;
    map['from'] = ugly_move.fromAlgebraic;
    map['captured'] = ugly_move.captured;

    var flags = '';
    for (var flag in BITS.keys) {
      if ((BITS[flag]! & ugly_move.flags) != 0) {
        flags += FLAGS[flag]!;
      }
    }
    map['flags'] = flags;

    return map;
  }

  String trim(String str) {
    return str.replaceAll(RegExp(r'^\s+|\s+$'), '');
  }

  int perft(int? depth) {
    var moves = generate_moves({'legal': false});
    var nodes = 0;
    var color = turn;

    // Capture the hash BEFORE making any moves at this level
    final int startKey = zobristKey;

    for (var i = 0, len = moves.length; i < len; i++) {
      final move = moves[i];
      make_move(move);

      if (!king_attacked(color)) {
        // --- ZOBRIST DEBUG CHECK ---
        final int expectedKey = generateZobristKey();
        if (zobristKey != expectedKey) {
          // Construct a simple algebraic string for the error message
          final moveStr =
              "${move.fromAlgebraic}${move.toAlgebraic}${move.promotion != null ? move.promotion!.name : ''}";

          throw Exception(
            'Zobrist mismatch after make_move!\n'
            'Move: $moveStr\n'
            'Incremental: $zobristKey\n'
            'From Scratch: $expectedKey',
          );
        }

        if (depth! - 1 > 0) {
          nodes += perft(depth - 1);
        } else {
          nodes++;
        }
      }

      undo_move();

      // --- UNDO CHECK ---
      if (zobristKey != startKey) {
        final moveStr = "${move.fromAlgebraic}${move.toAlgebraic}";
        throw Exception(
          'Zobrist mismatch after undo_move!\n'
          'Move: $moveStr\n'
          'Restored: $zobristKey\n'
          'Original: $startKey',
        );
      }
    }

    return nodes;
  }
  //Public APIs

  ///  Returns a list of legals moves from the current position.
  ///  The function takes an optional parameter which controls the
  ///  single-square move generation and verbosity.
  ///
  ///  The piece, captured, and promotion fields contain the lowercase
  ///  representation of the applicable piece.
  ///
  ///  The flags field in verbose mode may contain one or more of the following values:
  ///
  ///  'n' - a non-capture
  ///  'b' - a pawn push of two squares
  ///  'e' - an en passant capture
  ///  'c' - a standard capture
  ///  'p' - a promotion
  ///  'k' - kingside castling
  ///  'q' - queenside castling
  ///  A flag of 'pc' would mean that a pawn captured a piece on the 8th rank and promoted.
  ///
  ///  If "asObjects" is set to true in the options Map, then it returns a List<Move>
  List moves([Map? options]) {
    /* The internal representation of a chess move is in 0x88 format, and
       * not meant to be human-readable.  The code below converts the 0x88
       * square coordinates to algebraic coordinates.  It also prunes an
       * unnecessary move keys resulting from a verbose call.
       */

    final ugly_moves = generate_moves(options);
    if (options != null &&
        options.containsKey('asObjects') &&
        options['asObjects'] == true) {
      return ugly_moves;
    }
    final moves = [];

    for (var i = 0, len = ugly_moves.length; i < len; i++) {
      /* does the user want a full move object (most likely not), or just
         * SAN
         */
      if (options != null &&
          options.containsKey('verbose') &&
          options['verbose'] == true) {
        moves.add(make_pretty(ugly_moves[i]));
      } else {
        moves.add(move_to_san(ugly_moves[i]));
      }
    }

    return moves;
  }

  bool get in_draw {
    return half_moves >= 100 ||
        in_stalemate ||
        insufficient_material ||
        in_threefold_repetition;
  }

  bool get game_over {
    return in_draw || in_checkmate;
  }

  String get fen {
    return generate_fen();
  }

  /// return the san string representation of each move in history. Each string corresponds to one move.
  List<String?> san_moves() {
    /* pop all of history onto reversed_history */
    final reversed_history = <Move?>[];
    while (history.isNotEmpty) {
      reversed_history.add(undo_move());
    }

    var start_move_number = 1;
    if (header['FEN'] != null) {
      final move_number_string = header['FEN'].split(' ')[5];
      start_move_number = int.parse(move_number_string);
    }

    final moves = <String?>[];
    var move_string = '';
    var pgn_move_number = start_move_number;

    /* build the list of moves.  a move_string looks like: "3. e3 e6" */
    while (reversed_history.isNotEmpty) {
      final move = reversed_history.removeLast()!;

      /* if the position started with black to move, start PGN with ${start_move_number}. ... */
      if (pgn_move_number == start_move_number && move.color == BLACK) {
        move_string = '$start_move_number. ...';
        pgn_move_number++;
      } else if (move.color == WHITE) {
        /* store the previous generated move_string if we have one */
        if (move_string.isNotEmpty) {
          moves.add(move_string);
        }
        move_string = pgn_move_number.toString() + '.';
        pgn_move_number++;
      }

      move_string = move_string + ' ' + move_to_san(move);
      make_move(move);
    }

    /* are there any other leftover moves? */
    if (move_string.isNotEmpty) {
      moves.add(move_string);
    }

    /* is there a result? */
    if (header['Result'] != null) {
      moves.add(header['Result']);
    }

    return moves;
  }

  /// Return the PGN representation of the game thus far
  String pgn([Map? options]) {
    /* using the specification from http://www.chessclub.com/help/PGN-spec
       * example for html usage: .pgn({ max_width: 72, newline_char: "<br />" })
       */
    final newline =
        (options != null &&
            options.containsKey('newline_char') &&
            options['newline_char'] != null)
        ? options['newline_char']
        : '\n';
    final max_width =
        (options != null &&
            options.containsKey('max_width') &&
            options['max_width'] != null)
        ? options['max_width']
        : 0;
    final result = [];
    var header_exists = false;

    /* add the PGN header headerrmation */
    for (var i in header.keys) {
      /* TODO: order of enumerated properties in header object is not
         * guaranteed, see ECMA-262 spec (section 12.6.4)
         */
      result.add(
        '[' + i.toString() + ' \"' + header[i].toString() + '\"]' + newline,
      );
      header_exists = true;
    }

    if (header_exists && (history.isNotEmpty)) {
      result.add(newline);
    }

    final moves = san_moves();

    if (max_width == 0) {
      return result.join('') + moves.join(' ');
    }

    /* wrap the PGN output at max_width */
    var current_width = 0;
    for (var i = 0; i < moves.length; i++) {
      /* if the current move will push past max_width */
      if (current_width + moves[i]!.length > max_width && i != 0) {
        /* don't end the line with whitespace */
        if (result[result.length - 1] == ' ') {
          result.removeLast();
        }

        result.add(newline);
        current_width = 0;
      } else if (i != 0) {
        result.add(' ');
        current_width++;
      }
      result.add(moves[i]);
      current_width += moves[i]!.length;
    }

    return result.join('');
  }

  /// Load the moves of a game stored in Portable Game Notation.
  /// [options] is an optional parameter that contains a 'newline_char'
  /// which is a string representation of a RegExp (and should not be pre-escaped)
  /// and defaults to '\r?\n').
  /// Returns [true] if the PGN was parsed successfully, otherwise [false].
  bool load_pgn(String? pgn, [Map? options]) {
    String mask(str) {
      return str.replaceAll(RegExp(r'\\'), '\\');
    }

    /* convert a move from Standard Algebraic Notation (SAN) to 0x88
       * coordinates
      */
    Move? move_from_san(move) {
      final moves = generate_moves();
      for (var i = 0, len = moves.length; i < len; i++) {
        /* strip off any trailing move decorations: e.g Nf3+?! */
        if (move.replaceAll(RegExp(r'[+#?!=]+$'), '') ==
            move_to_san(moves[i]).replaceAll(RegExp(r'[+#?!=]+$'), '')) {
          return moves[i];
        }
      }
      return null;
    }

    Move? get_move_obj(move) {
      return move_from_san(trim(move));
    }

    /*has_keys(object) {
        bool has_keys = false;
        for (var key in object) {
          has_keys = true;
        }
        return has_keys;
      }*/

    Map<String, String> parse_pgn_header(header, [Map? options]) {
      final newline_char =
          (options != null && options.containsKey('newline_char'))
          ? options['newline_char']
          : '\r?\n';
      final header_obj = <String, String>{};
      final headers = header.split(RegExp(newline_char));
      var key = '';
      var value = '';

      for (var i = 0; i < headers.length; i++) {
        var keyMatch = RegExp(r'^\[([A-Z][A-Za-z]*)\s.*\]$');
        var temp = keyMatch.firstMatch(headers[i]);
        if (temp != null) {
          key = temp[1]!;
        }
        //print(key);
        var valueMatch = RegExp(r'^\[[A-Za-z]+\s"(.*)"\]$');
        temp = valueMatch.firstMatch(headers[i]);
        if (temp != null) {
          value = temp[1]!;
        }
        //print(value);
        if (trim(key).isNotEmpty) {
          header_obj[key] = value;
        }
      }

      return header_obj;
    }

    final newline_char =
        (options != null && options.containsKey('newline_char'))
        ? options['newline_char']
        : '\r?\n';
    //var regex = new RegExp(r'^(\[.*\]).*' + r'1\.'); //+ r"1\."); //+ mask(newline_char));

    final indexOfMoveStart = pgn!.indexOf(RegExp(newline_char + r'\d+\.{1,3}'));

    /* get header part of the PGN file */
    String? header_string;
    if (indexOfMoveStart != -1) {
      header_string = pgn.substring(0, indexOfMoveStart).trim();
    }

    /* no info part given, begins with moves */
    if (header_string == null || header_string[0] != '[') {
      header_string = '';
    }

    /* parse PGN header */
    final headers = parse_pgn_header(header_string, options);
    if (headers.containsKey('FEN')) {
      load(headers['FEN']!);
    } else {
      reset();
    }
    for (var key in headers.keys) {
      set_header([key, headers[key]]);
    }

    /* delete header to get the moves */
    var ms = pgn
        .replaceAll(header_string, '')
        .replaceAll(RegExp(mask(newline_char)), ' ');

    /* delete comments */
    ms = ms.replaceAll(RegExp(r'({[^}]+\})+?'), '');

    /* delete move numbers */
    ms = ms.replaceAll(RegExp(r'\d+\.{1,3}'), '');

    /* delete recursive annotation variations */
    RegExp regExp = RegExp(r'(\([^\(\)]+\))+?');
    var variations = regExp.allMatches(ms).toList();
    while (variations.isNotEmpty) {
      ms = ms.replaceAll(regExp, '');
      variations = regExp.allMatches(ms).toList();
    }

    /* trim and get array of moves */
    var moves = trim(ms).split(RegExp(r'\s+'));

    /* delete empty entries */
    moves = moves.join(',').replaceAll(RegExp(r',,+'), ',').split(',');
    var move;

    for (var half_move = 0; half_move < moves.length - 1; half_move++) {
      move = get_move_obj(moves[half_move]);

      /* move not possible! (don't clear the board to examine to show the
         * latest valid position)
         */
      if (move == null) {
        return false;
      } else {
        make_move(move);
      }
    }

    /* examine last move */
    move = moves[moves.length - 1];
    if (POSSIBLE_RESULTS.contains(move)) {
      if (!header.containsKey('Result')) {
        set_header(['Result', move]);
      }
    } else {
      final moveObj = get_move_obj(move);
      if (moveObj == null) {
        return false;
      } else {
        make_move(moveObj);
      }
    }
    return true;
  }

  /// The move function can be called with in the following parameters:
  /// .move('Nxb7')      <- where 'move' is a case-sensitive SAN string
  /// .move({ from: 'h7', <- where the 'move' is a move object (additional
  ///      to :'h8',      fields are ignored)
  ///      promotion: 'q',
  ///      })
  /// or it can be called with a Move object
  /// It returns true if the move was made, or false if it could not be.
  bool move(move) {
    Move? move_obj;
    final moves = generate_moves();

    if (move is String) {
      /* convert the move string to a move object */
      for (var i = 0; i < moves.length; i++) {
        if (move == move_to_san(moves[i])) {
          move_obj = moves[i];
          break;
        }
      }
    } else if (move is Map) {
      /* convert the pretty move object to an ugly move object */
      for (var i = 0; i < moves.length; i++) {
        if (move['from'] == moves[i].fromAlgebraic &&
            move['to'] == moves[i].toAlgebraic &&
            (moves[i].promotion == null ||
                move['promotion'] == moves[i].promotion!.name)) {
          move_obj = moves[i];
          break;
        }
      }
    } else if (move is Move) {
      move_obj = move;
    }

    /* failed to find move */
    if (move_obj == null) {
      return false;
    }

    /* need to make a copy of move because we can't generate SAN after the
       * move is made
       */

    make_move(move_obj);

    return true;
  }

  /// Takeback the last half-move, returning a move Map if successful, otherwise null.
  Map<String, dynamic>? undo() {
    final move = undo_move();
    return (move != null) ? make_pretty(move) : null;
  }

  /// Returns the color of the square ('light' or 'dark'), or null if [square] is invalid
  String? square_color(square) {
    if (SQUARES.containsKey(square)) {
      final sq_0x88 = SQUARES[square];
      return ((rank(sq_0x88) + file(sq_0x88)) % 2 == 0) ? 'light' : 'dark';
    }

    return null;
  }

  List getHistory([Map? options]) {
    final reversed_history = <Move?>[];
    final move_history = [];
    final verbose =
        (options != null &&
        options.containsKey('verbose') &&
        options['verbose'] == true);

    while (history.isNotEmpty) {
      reversed_history.add(undo_move());
    }

    while (reversed_history.isNotEmpty) {
      final move = reversed_history.removeLast()!;
      if (verbose) {
        move_history.add(make_pretty(move));
      } else {
        move_history.add(move_to_san(move));
      }
      make_move(move);
    }

    return move_history;
  }

  // Inside class Chess {
  static final List<List<int>> _zobristPieces = List.generate(
    128,
    (_) => List.filled(12, 0),
  );
  static final List<int> _zobristCastling = List.filled(
    16,
    0,
  ); // 4 bits = 16 states
  static final List<int> _zobristEnPassant = List.filled(128, 0);
  static late final int _zobristTurn;
  static bool _zobristInitialized = false;

  static void _initZobrist() {
    if (_zobristInitialized) return;

    // Using a fixed seed for deterministic behavior and easier debugging
    final random = Random(42);
    int next64() {
      // Dart's nextInt is 32-bit; we combine two for a 64-bit hash
      return (random.nextInt(1 << 32) << 32) | random.nextInt(1 << 32);
    }

    for (var i = 0; i < 128; i++) {
      for (var j = 0; j < 12; j++) {
        _zobristPieces[i][j] = next64();
      }
      _zobristEnPassant[i] = next64();
    }

    for (var i = 0; i < 16; i++) {
      _zobristCastling[i] = next64();
    }

    _zobristTurn = next64();
    _zobristInitialized = true;
  }

  int zobristKey = 0;

  // int generateZobristKey() {
  //   _initZobrist();
  //   int key = 0;

  //   // 1. Pieces (Standard 0x88 loop)
  //   for (var i = 0; i < 128; i++) {
  //     if ((i & 0x88) == 0 && board[i] != null) {
  //       key ^= _getPieceHash(board[i]!, i);
  //     }
  //   }

  //   // 2. Turn
  //   if (turn == BLACK) key ^= _zobristTurn;

  //   // 3. En Passant
  //   if (ep_square != EMPTY && ep_square != null) {
  //     key ^= _zobristEnPassant[ep_square!];
  //   }

  //   // 4. FIX: Normalize Castling Rights
  //   // We mask with 0x0F (binary 1111) to ensure the index stays 0-15.
  //   // If your 'castling' values are larger flags, you need to map them to 0-3.
  //   int whiteCastling = castling[WHITE] & 3; // Keep only the bottom 2 bits
  //   int blackCastling = castling[BLACK] & 3; // Keep only the bottom 2 bits

  //   int castlingIndex = (whiteCastling << 2) | blackCastling;

  //   key ^= _zobristCastling[castlingIndex];

  //   return key;
  // }

  int _getCastlingIndex() {
    int index = 0;
    // Map your specific bitflags (32 and 64) to a clean 4-bit number (0-15)
    if ((castling[WHITE] & BITS_KSIDE_CASTLE) != 0) index |= 1;
    if ((castling[WHITE] & BITS_QSIDE_CASTLE) != 0) index |= 2;
    if ((castling[BLACK] & BITS_KSIDE_CASTLE) != 0) index |= 4;
    if ((castling[BLACK] & BITS_QSIDE_CASTLE) != 0) index |= 8;
    return index;
  }

  int generateZobristKey() {
    _initZobrist();
    int key = 0;

    // 1. Pieces
    for (var i = 0; i < 128; i++) {
      if ((i & 0x88) == 0 && board[i] != null) {
        key ^= _getPieceHash(board[i]!, i);
      }
    }

    // 2. Turn
    if (turn == BLACK) key ^= _zobristTurn;

    // 3. En Passant
    if (ep_square != EMPTY && ep_square != null) {
      key ^= _zobristEnPassant[ep_square!];
    }

    // 4. Castling (Using the helper)
    key ^= _zobristCastling[_getCastlingIndex()];

    return key;
  }

  int _getPieceHash(Piece piece, int square) {
    // Standard mapping: P=0, N=1, B=2, R=3, Q=4, K=5
    // You can use a Map or a simple switch to ensure typeIdx is 0-5
    int typeIdx;
    switch (piece.type) {
      case PAWN:
        typeIdx = 0;
        break;
      case KNIGHT:
        typeIdx = 1;
        break;
      case BISHOP:
        typeIdx = 2;
        break;
      case ROOK:
        typeIdx = 3;
        break;
      case QUEEN:
        typeIdx = 4;
        break;
      case KING:
        typeIdx = 5;
        break;
      default:
        typeIdx = 0;
    }

    // Offset for Black pieces (6-11)
    if (piece.color == BLACK) {
      typeIdx += 6;
    }

    return _zobristPieces[square][typeIdx];
  }
}

class Piece {
  PieceType type;
  final Color color;
  Piece(this.type, this.color);
}

class PieceType {
  final int shift;
  final String name;
  const PieceType._internal(this.shift, this.name);

  static const PieceType PAWN = PieceType._internal(0, 'p');
  static const PieceType KNIGHT = PieceType._internal(1, 'n');
  static const PieceType BISHOP = PieceType._internal(2, 'b');
  static const PieceType ROOK = PieceType._internal(3, 'r');
  static const PieceType QUEEN = PieceType._internal(4, 'q');
  static const PieceType KING = PieceType._internal(5, 'k');

  @override
  String toString() => name;
  String toLowerCase() => name;
  String toUpperCase() => name.toUpperCase();
}

enum Color { WHITE, BLACK }

class ColorMap<T> {
  T _white;
  T _black;
  ColorMap(T value) : _white = value, _black = value;
  ColorMap.clone(ColorMap other) : _white = other._white, _black = other._black;

  T operator [](Color color) {
    return (color == Color.WHITE) ? _white : _black;
  }

  void operator []=(Color color, T value) {
    if (color == Color.WHITE) {
      _white = value;
    } else {
      _black = value;
    }
  }
}

class Move {
  final Color color;
  final int from;
  final int to;
  final int flags;
  final PieceType piece;
  final PieceType? captured;
  final PieceType? promotion;
  const Move(
    this.color,
    this.from,
    this.to,
    this.flags,
    this.piece,
    this.captured,
    this.promotion,
  );

  String get fromAlgebraic {
    return Chess.algebraic(from);
  }

  String get toAlgebraic {
    return Chess.algebraic(to);
  }
}

class State {
  final Move move;
  final ColorMap<int> kings;
  final Color turn;
  final ColorMap<int> castling;
  final int? ep_square;
  final int half_moves;
  final int move_number;
  final int
  zobristKey; // Added to store the hash of the position BEFORE the move

  State(
    this.move,
    this.kings,
    this.turn,
    this.castling,
    this.ep_square,
    this.half_moves,
    this.move_number,
    this.zobristKey,
  );
}
