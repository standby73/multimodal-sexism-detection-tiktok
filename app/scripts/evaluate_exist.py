"""Refactored mechanically from a notebook (payoevall.ipynb). Not re-run after conversion."""


def vote_to_label(votes):
    """
    Convierte votos binarios a etiqueta {'YES','NO'} con desempate determinista.
    Regla igual a argmax sobre [count_NO, count_YES]: en empate gana NO.
    """
    if not votes:
        return None

    no_count = sum(1 for v in votes if int(v) == 0)
    yes_count = sum(1 for v in votes if int(v) == 1)
    return "YES" if yes_count > no_count else "NO"


from maikol_utils.file_utils import load_json

pred_path = '../data/processed/predictions/xlm-roberta-base_test_loaded_model.json'
ref_path = '../data/processed/test_payo.json'

test_data = load_json('../data/processed/test.json')
preds = load_json(pred_path)

# Referencia determinista alineada con la lógica usada en evaluate (argmax en empate -> NO).
test = [
    {
        'id': str(s['id']),
        'test_case': 'EXIST2026',
        'value': vote_to_label(s['label'])
    }
    for s in test_data
]

# Chequeos de consistencia antes de evaluar
pred_ids = [str(x['id']) for x in preds]
ref_ids = [str(x['id']) for x in test]
print(f'pred_path={pred_path}')
print(f'n_preds={len(preds)} | n_ref={len(test)}')
print(f'ids_orden_igual={pred_ids == ref_ids}')
print(f'ids_set_igual={set(pred_ids) == set(ref_ids)}')


from maikol_utils.file_utils import save_json
save_json('../data/processed/test_payo.json', test)


from pyevall.evaluation import PyEvALLEvaluation
from pyevall.metrics.metricfactory import MetricFactory
from itertools import combinations

test = PyEvALLEvaluation()
ref_path = '../data/processed/test_payo.json'

pred_path = [
    '../data/processed/predictions/send/turboabuelas_trainedonboth.json',
'../data/processed/predictions/send/turboabuelas_trainedonenglish.json', 
 '../data/processed/predictions/send/turboabuelas_trainedonspanish.json' 
]

for f in pred_path:
    print(f'=========================================================')
    print(f'Evaluando predicciones de: {f}')
    print(f'=========================================================')

    files = [
        f,
        ref_path,
    ]

    metrics = [
        MetricFactory.Accuracy.value,
        MetricFactory.FMeasure.value,
    ]

    params = dict()

    for preds_file, labels_file in combinations(files, 2):
        print(f"\nEvaluando:\nPreds: {preds_file}\nLabels: {labels_file}\n")
        report = test.evaluate(preds_file, labels_file, metrics, **params)
        report.print_report()


from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils
from pyevall.metrics.metricfactory import MetricFactory
test = PyEvALLEvaluation()
labels = '../output/TurboAbuelas_RobertMostro.json'
preds = '../output/TurboAbuelas_Qwen.json'



metrics = [
 MetricFactory.Accuracy.value,
 MetricFactory.FMeasure.value,
]
params= dict()
report = test.evaluate(preds, labels, metrics, **params)
report.print_report()
# report.report, if you want to access the dictionary :)
