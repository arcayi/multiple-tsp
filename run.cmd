@echo ���ݵ�ǰ����ʱ�䣬�����ļ����ƣ�......
@set YYYYmmdd=%date:~0,4%%date:~5,2%%date:~8,2%
@set hhmiss=%time:~0,2%%time:~3,2%%time:~6,2%
@set "filename=cProfile_%YYYYmmdd%_%hhmiss%.out"
@echo %filename%

python -m cProfile -o .out\%filename% ga-mtsp.py

@REM python -c "import pstats; p=pstats.Stats('.out\%filename%'); p.sort_stats('cumulative').print_stats()"

gprof2dot -n 1 -e .2 -f pstats .out\%filename% | dot -Tpng -o .out\%filename%.png