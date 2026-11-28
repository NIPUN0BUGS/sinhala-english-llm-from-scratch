# Training data sources and collection policy

The scheduled GitHub Actions workflow collects data on GitHub-hosted runners. It does not require the developer's local PC to be running.

## Reviewed sources

| Source | Language | Published licence | Use in the collector |
|---|---|---|---|
| [Minuri NSINA sentence corpus](https://huggingface.co/datasets/Minuri/nsina-sentences-raw) | Sinhala | CC BY-SA 4.0 | Text rows from the `train` split |
| [Sri Lanka Parliament Hansard trilingual dataset](https://huggingface.co/datasets/keshan/lk_hansard_trilingual) | Sinhala and English | CC BY 4.0 | Sinhala and English splits |
| [Sinhala Wikipedia](https://si.wikipedia.org/) and [English Wikipedia](https://en.wikipedia.org/) | Sinhala and English | CC BY-SA 4.0 | Smaller top-up collection |

The reviewed sources are accessed through the Hugging Face Dataset Viewer API. The collector fetches bounded batches of at most 100 rows per API request, uses eight concurrent requests by default, checks the expected language script, normalizes text, and removes exact normalized-text duplicates. It records the dataset, split, row identifier, source metadata, attribution, source URL and licence on each manifest entry.

## Attribution and licence handling

- CC BY 4.0 records retain a link to the CC BY 4.0 licence and the dataset/source attribution.
- CC BY-SA 4.0 records retain a link to the CC BY-SA 4.0 licence and the dataset/source attribution.
- Each record retains its own licence. Do not combine records into a redistributed dataset without preserving the applicable attribution and ShareAlike obligations.
- The collector only accepts the reviewed sources and licence identifiers in `src/collect_open_datasets.py`. New sources must be reviewed before being added.
- Dataset availability and API support can change. A source that is unavailable is skipped with a log message; the existing corpus is not deleted.

## Sources deliberately not used

The LIRNEasia Facebook Decade Corpora page describes a CC BY 4.0 release, but its repository README marks the large corpora as deprecated and removed from public access. The pipeline therefore does not assume those files can be downloaded. The CulturaX-derived Sinhala corpus was deliberately excluded: its dataset card states that the upstream mC4/OSCAR source requires a Hugging Face contact-information agreement. We use the separately documented NSINA-derived source instead. Large mixed-source corpora are also not added until their component licences and attribution requirements can be retained safely.

## Speed and target

The collector uses batched dataset API requests rather than waiting between individual random Wikipedia page requests. Eight concurrent requests are used by default. This is an engineering throughput improvement, not a promise of exactly 20x speed: actual speed depends on dataset-server availability, rate limits, response sizes and GitHub Actions capacity.

The workflow measures the resulting manifest using the project's current character-level tokenizer metric. It does not begin the gated self-learning training step until the total reaches 1,000,000 Unicode character-level tokens and each language reaches at least 100,000 tokens.
