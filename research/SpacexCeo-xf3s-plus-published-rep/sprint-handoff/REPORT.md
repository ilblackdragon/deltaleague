# Delta League sprint 05.11.2026 — сводный отчёт

## Итог
- Сабмит: `51721d34-391e-4c58-bc5d-57bccff5016e` — https://deltaleague.ai/submissions/51721d34-391e-4c58-bc5d-57bccff5016e/ — VERIFIED: private 33.0, validation 41.44, truncation 0.15, место 4 (лидер xf3s 36.0)
- Research PR: https://github.com/ilblackdragon/deltaleague/pull/24 (ветка SpacexCeo/deltaleague:claude/compassionate-albattani-fdhmk0)
- Правка: родитель b8af1dbc (xf3s, 36.0% private) + опубликованная правка повторов (Sparks of Weight-Space Programming, слои 15/19 MLP down_proj) с силой ×0.5. Формат sparse-diff, 2 тензора.
- Квота: использован 1 сабмит из 2.

## Локальные результаты (прокси, советующие, не официальные)
| | xf3s (родитель) | наш кандидат ×0.5 |
|---|---|---|
| strict-точность (2 набора сидов, 1920 генераций) | 43.2 | 45.6 (+2.4 ± 1.2) |
| доля обрезанных ответов | 0.28 | 0.18 |
| стресс-набор 265 задач | 40.0 | 45.3 |
Прогноз private по калибровке A1: ≈ 37.4% (80% интервал 34.6–40.1). Строгий порог «неубиваемости» (+4.1 strict) не пройден — отправлено как проверка гипотезы.

## Что не сработало (поверх xf3s)
- `</think>` ×0.92 / ×0.96: обмен «разросшихся ответов» на «незакрытые», в среднем ≈ 0.
- gain ×0.9 / ×0.95: нейтрально. Защита от закрытия на первом токене: −1.4. Направление «преждевременного закрытия»: −6.4.
- Правка повторов ×1.0: +1.1 strict, но ломает MMLU на стресс-наборе (50 → 27.5).

## Отчёты агентов
- A1 калибровка и правило отбора: /home/user/delta/agents/a1/REPORT.md
- A2 токены и запас закрытия: /home/user/delta/agents/a2/REPORT.md
- A3 математика сэмплера: /home/user/delta/agents/a3/REPORT.md
- A4 динамика закрытия `</think>`: /home/user/delta/agents/a4/REPORT.md
- A5 направления в скрытых состояниях, правка повторов: /home/user/delta/agents/a5/REPORT.md
- A6 сетка и строгая метрика: /home/user/delta/agents/a6/REPORT.md (таблица: /home/user/delta/agents/a6/grid.csv)
- F1 (Fable) общая модель: /home/user/delta/agents/f1/REPORT.md
- Общий бриф агентов (все факты и обновления): /home/user/delta/AGENT_BRIEF.md
- Research-запись в PR: /home/user/deltaleague/research/SpacexCeo-xf3s-plus-published-rep/README.md

## Официальный результат и вывод
- Private 33.0 (лидер xf3s 36.0), validation 41.44, доля обрезанных 0.15 (у xf3s 0.23). Место 4 на доске.
- Механизм подтвердился: обрезок стало меньше. Но точность закончившихся ответов упала, и private снизился на 3.0
  (примерно 1 SE разницы). Это ровно то, о чём предупреждал guard A1 «точность среди нормально закрытых −5».
- Вывод: на стеке xf3s private награждает качество ответа, а не долю закрытых рассуждений. Следующие правки должны
  повышать точность закрытых ответов, а не их количество; guard acc|healthy делать жёстким фильтром.
