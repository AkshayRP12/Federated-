import sqlite3
conn = sqlite3.connect("results/stage4_grid.sqlite")
cur = conn.cursor()
for row in cur.execute("SELECT domain, COUNT(*), COUNT(DISTINCT scenario_id), COUNT(DISTINCT config_id) FROM experiment_runs GROUP BY domain"):
    print(f"Domain {row[0]}: {row[1]} runs across {row[2]} scenarios and {row[3]} configs")
conn.close()
