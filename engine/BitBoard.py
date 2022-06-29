import  chess


def generateBitBoard(full_fen):
    splitted_fen = full_fen.split(' ')
    fen = splitted_fen[0]
    board = chess.Board(fen)
    board.pieces(chess.KING,chess.WHITE)
    endString = ''
    for color in chess.COLORS:
        for pieceType in chess.PIECE_TYPES:
            for x in str(board.pieces(pieceType,color)):
                if x == '.':
                    endString += '0'
                if x == '1':
                    endString += '1'
    if splitted_fen[1] == 'w':
        whiteMove = 1
    else:
        whiteMove = 0
    endString += str(whiteMove)
    enPassants = ['a3','b3','c3','d3','e3','f3','g3','h3','a6','b6','c6','d6','e6','f6','g6','h6']
    for enPassant in enPassants:
        if splitted_fen[2] == enPassant:
            endString += '1'
        else:
            endString += '0'
    castles = ['K','Q','k','q']
    for castle in castles:
        if castle in splitted_fen[3]:
            endString += '1'
        else:
            endString += '0'


generateBitBoard('rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b d3 KQkq')