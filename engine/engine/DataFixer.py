import sqlite3
from BitBoard import generate_bit_board

con = sqlite3.connect("test.db")
con2 = sqlite3.connect("example.db")
cur2 = con2.cursor()
cur = con.cursor()
cur.execute("select * from evaluations")
i = 0
# 63679
for row in cur:
    # print(row)
    id = str(row[0])
    fen = row[1]
    evaluation = float(row[3])
    bb = generate_bit_board(fen)
    con2.execute("insert into chess_table values ('"+fen + "', '" + bb + "', '" + str(evaluation) +"')")
    i += 1
    if i == 10000000:
        break
    print(i)
con2.commit()
