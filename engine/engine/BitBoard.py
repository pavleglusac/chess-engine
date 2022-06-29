import chess

def generate_bit_board(full_fen):
    splitted_fen = full_fen.split(' ')
    fen = splitted_fen[0]
    board = chess.Board(fen)
    board.pieces(chess.KING, chess.WHITE)
    end_string = ''
    for color in chess.COLORS:
        for pieceType in chess.PIECE_TYPES:
            for x in str(board.pieces(pieceType, color)):
                if x == '.':
                    end_string += '0'
                if x == '1':
                    end_string += '1'
    if splitted_fen[1] == 'w':
        white_move = 1
    else:
        white_move = 0
    end_string += str(white_move)
    en_passants = ['a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3', 'a6', 'b6', 'c6', 'd6', 'e6', 'f6', 'g6', 'h6']
    for en_passant in en_passants:
        if splitted_fen[2] == en_passant:
            end_string += '1'
        else:
            end_string += '0'
    castles = ['K', 'Q', 'k', 'q']
    for castle in castles:
        if castle in splitted_fen[3]:
            end_string += '1'
        else:
            end_string += '0'
    return end_string


if __name__ == '__main__':
    ret = generate_bit_board('rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b d3 KQkq')
    print(ret)
