import math

import chess
import copy
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from engine.engine.Model import ChessEngine

from engine.engine.Tree import Tree, TreeNode
from engine.engine.BitBoard import generate_bit_board

def getFirstXMoves(legal_moves,board,chess_engine,size):
    dict = {}
    for legal_move in legal_moves:
        board_copy = copy.deepcopy(board)
        board_copy.push_san(str(legal_move))
        dict[legal_move] = chess_engine.evaluate_fen(board_copy.fen())
    touple_list = sorted(dict.items(), key=lambda x: x[1], reverse = board.fen().split(' ')[1] != 'w')
    return  [x[0] for x in touple_list][0:size]


def playMove(fen, depth, chess_engine):
    print(fen)
    tree = Tree()
    tree.root = TreeNode(0)
    board = chess.Board(fen)
    dict = {}
    for item in getFirstXMoves(list(board.legal_moves), board, chess_engine, 5):
        print(item)
        dict[item] = minimax(copy.deepcopy(board), item, depth - 1, tree.root, -math.inf, math.inf, chess_engine)
    touple_list = sorted(dict.items(), key=lambda x: x[1], reverse = board.fen().split(' ')[1] != 'w')
    return  [x[0] for x in touple_list][0]


def minimax(board, legal_move, depth, parent, alpha, beta, chess_engine):
    if depth == 0:
        return chess_engine.evaluate_fen(fen=board.fen())
    board.push_san(str(legal_move))
    child = TreeNode(0)
    parent.add_child(child)
    if board.fen().split(' ')[1] == 'w':
        bestVal = -math.inf
        for legal_movee in getFirstXMoves(list(board.legal_moves), board, chess_engine, 5):
            val = minimax(copy.deepcopy(board), legal_movee, depth - 1, child, alpha, beta, chess_engine)
            bestVal = max(bestVal, val)
            alpha = max(alpha, bestVal)
            if beta <= alpha:
                break
    else:
        bestVal = math.inf
        for legal_movee in getFirstXMoves(list(board.legal_moves), board, chess_engine, 5):
            val = minimax(copy.deepcopy(board), legal_movee, depth - 1, child, alpha, beta, chess_engine)
            bestVal = min(bestVal, val)
            beta = min(beta, bestVal)
            if beta <= alpha:
                break
    child.data = bestVal
    return bestVal


if __name__ == "__main__":
    import time

    start_time = time.time()
    chess_engine = ChessEngine()
    print(playMove("rn3rk1/ppp3pp/4pn2/1q6/8/P2P1N2/1PPQBPPP/R4RK1 w - - 0 13", 2,chess_engine))
    print("My program took", time.time() - start_time, "to run")
