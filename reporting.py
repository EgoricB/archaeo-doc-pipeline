from pathlib import Path
import pandas as pd
from utils import save_json


def save_outputs(output_dir, manifest_rows, page_rows, entity_rows, corpus_stats_obj):
    output_dir = Path(output_dir)

    pd.DataFrame(manifest_rows).to_csv(output_dir / 'pages_manifest.csv', index=False)
    pd.DataFrame(page_rows).to_csv(output_dir / 'page_analysis.csv', index=False)
    pd.DataFrame(entity_rows).to_csv(output_dir / 'entities.csv', index=False)
    save_json(corpus_stats_obj, output_dir / 'corpus_stats.json')

    report = []
    report.append('# Report')
    report.append('')
    report.append(f'- Pages: {len(page_rows)}')
    report.append(f'- Entities: {len(entity_rows)}')
    report.append(f"- Documents in corpus stats: {corpus_stats_obj.get('num_documents', 0)}")
    report.append('')
    report.append('## Top tokens')
    report.append('')

    for tok, n in corpus_stats_obj.get('top_50_tokens', [])[:20]:
        report.append(f'- {tok}: {n}')

    with open(output_dir / 'report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))