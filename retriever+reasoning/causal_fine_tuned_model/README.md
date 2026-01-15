---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:7276
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) <!-- at revision c5ee24cb16019beea0893ab7796b1df96625c6b8 -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of texts
pairs = [
    ['The D.C. National Guard was activated, and a citywide curfew was imposed. | Trump supporters protested in Washington, D.C.', 'Pentagon faces deadline on recommending whether to invoke the Insurrection Act: Happy Friday! We are getting close to the 90-day deadline that President Donald Trump set back in January for the secretaries of the Departments of Defense and Homeland Security to recommend whether the president should invoke the Insurrection Act to address what he described as an “invasion” of gangs, human traffickers, and criminals at the southern border.\n\nMore than 10,000 service members are currently deployed to the U.S.-Mexico border. The Posse Comitatus Act of 1878 prevents federal troops Law enforcement missed key signs ahead of riot on US Capitol: CNN —\n\nDespite weeks of planning between federal and local police agencies ahead of Wednesday’s Trump rally – including tracking social media – officials said that going into Wednesday they had no intelligence indicating there was a threat the US Capitol could be overrun.\n\nIt would turn out to be a catastrophic failure after an angry mob overwhelmed police and stormed into the Capitol building, ransacking lawmakers’ offices, injuring dozens of officers and stealing electronics and documents that Trump Supporters Storm U.S. Capitol, Clash With Police : Capitol Insurrection Updates: Trump Supporters Storm U.S. Capitol, Clash With Police\n\nEnlarge this image toggle caption Joseph Prezioso/AFP via Getty Images Joseph Prezioso/AFP via Getty Images\n\nUpdated 3:08 p.m. ET\n\nThousands of Trump supporters stormed the U.S. Capitol on Wednesday, prompting the House and Senate to abruptly take a recess as the U.S. Capitol Police locked down the building. D.C. Mayor Muriel Bowser ordered a citywide curfew from 6 p.m. on Wednesday until 6 a.m. on Thursday.\n\nProtesters had amassed in Washi'],
    ["DeepSeek's app reached the top of U.S. free app charts within a week of its launch. | DeepSeek unveiled its V3 model in December and R1 model in January.", "DeepSeek R1 Surpasses ChatGPT and Google Gemini to Become the Top AI App Worldwide: Chinese technology company DeepSeek has made headlines globally with the launch of its open-source AI model, DeepSeek-R1. The model has rapidly gained popularity, igniting discussions across social media platforms and becoming a trending topic. Users have shared various memes and opinions, praising its capabilities and disruptive impact on the AI industry.\n\nDeepSeek-R1’s associated application has surged in downloads, becoming the most downloaded free AI app on the US Apple App Store. It achieve Chinese tech start-up DeepSeek unnerves US with low-costs AI model on par with OpenAI's o1: deepseek\n\nChinese tech start-up DeepSeek is shaking the US' AI sector with its open-source approach and low-cost models, drawing curiosity and significant coverage from US media and AI forums about how this Chinese company has produced competitive AI systems comparable to those developed by US tech giants, all while navigating the strict semiconductor restrictions imposed by the US government on China. The Global Times on Saturday talked to the company and several AI industrial observers to illu DeepSeek AI is a 'gift to the world': The biggest story out of China right now, here's why: The study, published in Nature Human Behaviour, was led by Michelle Vaccaro, Abdullah Almaatouq, and Thomas Malone (“When Combinations of Humans and AI Are Useful”) was the first large-scale meta-analysis conducted to better understand when human-AI combinations are useful in task completion, and when they are not."],
    ['Widespread disruption occurred in internet and news access, and banks temporarily shut down across Myanmar following the coup. | The election commission denied widespread voter fraud.', 'Myanmar’s military stages coup d’etat: Live news: In a series of morning raids, the military arrests senior government members and declares a state of emergency.\n\nMyanmar’s military has seized power and declared a state of emergency for one year following days of escalating tension over the result of November’s parliamentary elections.\n\nAung San Suu Kyi, the country’s de facto leader, President Win Myint and other senior members of the National League for Democracy (NLD) party have been detained in the capital, Naypyidaw, on Monday.\n\nA video br End of Myanmar’s Rocky Road to Democracy?: After days of speculation about an impending coup, Myanmar’s military has formally seized power the very day a newly elected parliament was scheduled to meet for the first time. Military generals ruled the country from the early 1960s until 2011. Now they are taking back control, after a near-decade of sharing power with elected lawmakers.\n\nPolitical Veto by Coup\n\nThe coup began in the early hours of February 1. The military detained senior politicians from Myanmar’s largest political party, the With a coup, Myanmar’s military upended its comfortable status quo. Why?: Myanmar’s military controls vast business interests in mining, telecommunications, textiles, hotels and even beer breweries and distribution. It faces no civilian oversight and can block changes to the country’s constitution. When it had to respond to international condemnation for its brutal repression of ethnic minorities, it trotted out members of its compliant civilian government to absorb the blame.\n\nWith such political and economic power, there seemed little for the military, known as the '],
    ['US President Donald Trump ordered the assassination of Iranian General Qassem Soleimani. | President Donald Trump withdrew the U.S. from the nuclear accord after taking office.', 'Trump recounts Qassem Soleimani\'s final moments: US President Donald Trump delivers remarks following the US military airstrike against Iranian General Qassem Soleimani in Baghdad, Iraq, in West Palm Beach, Florida, US, January 3, 2020\n\nTrump delivered the account Friday night to Republican Party donors gathered at his Florida residence Mar-a-Lago for a fundraising dinner, US media said.\n\nCNN on Saturday broadcast an audio recording in which the president gave new details about the January 3 strike at the airport in Baghdad. It killed the Revo Qassem Soleimani death: U.S. long watched Iranian general but feared the fallout of a strike: In 2007, U.S. commandos watched as a convoy carrying a powerful Iranian military leader made its way to northern Iraq. It was a prime opportunity to take out Gen. Qassem Soleimani, who had been accused of aiding Shiite forces that killed thousands of American troops in Iraq.\n\nBut ultimately, military leaders passed on a strike, deferring to deep concerns about the potential fallout of such a provocative attack.\n\n"To avoid a firefight, and the contentious politics that would follow, I decided tha World reacts to US killing of Iran’s Qassem Soleimani in Iraq: Leaders across the world warn that US’s targeted killing of Iranian top general could ignite conflict in region.\n\nInternational leaders have called for restraint and de-escalation following the assassination of Iran’s top general, ordered by US President Donald Trump, as Iran’s allies warned the killing could lead to conflict.\n\nQassem Soleimani, head of Iran’s elite Revolutionary Guard Corps (IRGC) Quds Force, was killed in a pre-dawn US air raid at Baghdad’s international airport on Friday.\n\nAt'],
    ["The US Senate will likely vote on Wednesday on a bill to halt Trump’s tariffs on Canadian imports. | Premier Ford paused Ontario's electricity surcharge.", "Donald Trump updates: World on edge as reciprocal tariffs deadline looms: Donald Trump updates: World on edge as reciprocal tariffs deadline looms\n\nUS Senate will likely vote on Wednesday on a bill to halt Trump’s tariffs on Canadian imports amid economic uncertainty. Trump will not impose 50% Canadian steel, aluminum tariffs tomorrow, says top trade advisor: President Donald Trump no longer plans to raise tariffs on Canadian steel and aluminum imports to 50% on Wednesday, top White House trade advisor Peter Navarro told CNBC on Tuesday afternoon. The reversal came six hours after Trump announced his plan to double import duties on the Canadian metals in response to Ontario's decision to slap a 25% tax on electricity exports to the U.S. Ontario Premier Doug Ford said earlier Tuesday afternoon that he was pausing that surcharge following discussions w Trump’s Tariffs Set Off Day of Anger, Retaliation and Market Unease: the new new world\n\nAs the United States grapples with the upheaval unleashed by the Trump administration, many Chinese people are finding they can relate to what many Americans are going through.\n\nThey are saying it feels something like the Cultural Revolution, the period known as “the decade of turmoil.” The young aides Elon Musk has sent to dismantle the U.S. government reminded some Chinese of the Red Guards whom Mao Zedong enlisted to destroy the bureaucracy at the peak of the Cultural Revol"],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'The D.C. National Guard was activated, and a citywide curfew was imposed. | Trump supporters protested in Washington, D.C.',
    [
        'Pentagon faces deadline on recommending whether to invoke the Insurrection Act: Happy Friday! We are getting close to the 90-day deadline that President Donald Trump set back in January for the secretaries of the Departments of Defense and Homeland Security to recommend whether the president should invoke the Insurrection Act to address what he described as an “invasion” of gangs, human traffickers, and criminals at the southern border.\n\nMore than 10,000 service members are currently deployed to the U.S.-Mexico border. The Posse Comitatus Act of 1878 prevents federal troops Law enforcement missed key signs ahead of riot on US Capitol: CNN —\n\nDespite weeks of planning between federal and local police agencies ahead of Wednesday’s Trump rally – including tracking social media – officials said that going into Wednesday they had no intelligence indicating there was a threat the US Capitol could be overrun.\n\nIt would turn out to be a catastrophic failure after an angry mob overwhelmed police and stormed into the Capitol building, ransacking lawmakers’ offices, injuring dozens of officers and stealing electronics and documents that Trump Supporters Storm U.S. Capitol, Clash With Police : Capitol Insurrection Updates: Trump Supporters Storm U.S. Capitol, Clash With Police\n\nEnlarge this image toggle caption Joseph Prezioso/AFP via Getty Images Joseph Prezioso/AFP via Getty Images\n\nUpdated 3:08 p.m. ET\n\nThousands of Trump supporters stormed the U.S. Capitol on Wednesday, prompting the House and Senate to abruptly take a recess as the U.S. Capitol Police locked down the building. D.C. Mayor Muriel Bowser ordered a citywide curfew from 6 p.m. on Wednesday until 6 a.m. on Thursday.\n\nProtesters had amassed in Washi',
        "DeepSeek R1 Surpasses ChatGPT and Google Gemini to Become the Top AI App Worldwide: Chinese technology company DeepSeek has made headlines globally with the launch of its open-source AI model, DeepSeek-R1. The model has rapidly gained popularity, igniting discussions across social media platforms and becoming a trending topic. Users have shared various memes and opinions, praising its capabilities and disruptive impact on the AI industry.\n\nDeepSeek-R1’s associated application has surged in downloads, becoming the most downloaded free AI app on the US Apple App Store. It achieve Chinese tech start-up DeepSeek unnerves US with low-costs AI model on par with OpenAI's o1: deepseek\n\nChinese tech start-up DeepSeek is shaking the US' AI sector with its open-source approach and low-cost models, drawing curiosity and significant coverage from US media and AI forums about how this Chinese company has produced competitive AI systems comparable to those developed by US tech giants, all while navigating the strict semiconductor restrictions imposed by the US government on China. The Global Times on Saturday talked to the company and several AI industrial observers to illu DeepSeek AI is a 'gift to the world': The biggest story out of China right now, here's why: The study, published in Nature Human Behaviour, was led by Michelle Vaccaro, Abdullah Almaatouq, and Thomas Malone (“When Combinations of Humans and AI Are Useful”) was the first large-scale meta-analysis conducted to better understand when human-AI combinations are useful in task completion, and when they are not.",
        'Myanmar’s military stages coup d’etat: Live news: In a series of morning raids, the military arrests senior government members and declares a state of emergency.\n\nMyanmar’s military has seized power and declared a state of emergency for one year following days of escalating tension over the result of November’s parliamentary elections.\n\nAung San Suu Kyi, the country’s de facto leader, President Win Myint and other senior members of the National League for Democracy (NLD) party have been detained in the capital, Naypyidaw, on Monday.\n\nA video br End of Myanmar’s Rocky Road to Democracy?: After days of speculation about an impending coup, Myanmar’s military has formally seized power the very day a newly elected parliament was scheduled to meet for the first time. Military generals ruled the country from the early 1960s until 2011. Now they are taking back control, after a near-decade of sharing power with elected lawmakers.\n\nPolitical Veto by Coup\n\nThe coup began in the early hours of February 1. The military detained senior politicians from Myanmar’s largest political party, the With a coup, Myanmar’s military upended its comfortable status quo. Why?: Myanmar’s military controls vast business interests in mining, telecommunications, textiles, hotels and even beer breweries and distribution. It faces no civilian oversight and can block changes to the country’s constitution. When it had to respond to international condemnation for its brutal repression of ethnic minorities, it trotted out members of its compliant civilian government to absorb the blame.\n\nWith such political and economic power, there seemed little for the military, known as the ',
        'Trump recounts Qassem Soleimani\'s final moments: US President Donald Trump delivers remarks following the US military airstrike against Iranian General Qassem Soleimani in Baghdad, Iraq, in West Palm Beach, Florida, US, January 3, 2020\n\nTrump delivered the account Friday night to Republican Party donors gathered at his Florida residence Mar-a-Lago for a fundraising dinner, US media said.\n\nCNN on Saturday broadcast an audio recording in which the president gave new details about the January 3 strike at the airport in Baghdad. It killed the Revo Qassem Soleimani death: U.S. long watched Iranian general but feared the fallout of a strike: In 2007, U.S. commandos watched as a convoy carrying a powerful Iranian military leader made its way to northern Iraq. It was a prime opportunity to take out Gen. Qassem Soleimani, who had been accused of aiding Shiite forces that killed thousands of American troops in Iraq.\n\nBut ultimately, military leaders passed on a strike, deferring to deep concerns about the potential fallout of such a provocative attack.\n\n"To avoid a firefight, and the contentious politics that would follow, I decided tha World reacts to US killing of Iran’s Qassem Soleimani in Iraq: Leaders across the world warn that US’s targeted killing of Iranian top general could ignite conflict in region.\n\nInternational leaders have called for restraint and de-escalation following the assassination of Iran’s top general, ordered by US President Donald Trump, as Iran’s allies warned the killing could lead to conflict.\n\nQassem Soleimani, head of Iran’s elite Revolutionary Guard Corps (IRGC) Quds Force, was killed in a pre-dawn US air raid at Baghdad’s international airport on Friday.\n\nAt',
        "Donald Trump updates: World on edge as reciprocal tariffs deadline looms: Donald Trump updates: World on edge as reciprocal tariffs deadline looms\n\nUS Senate will likely vote on Wednesday on a bill to halt Trump’s tariffs on Canadian imports amid economic uncertainty. Trump will not impose 50% Canadian steel, aluminum tariffs tomorrow, says top trade advisor: President Donald Trump no longer plans to raise tariffs on Canadian steel and aluminum imports to 50% on Wednesday, top White House trade advisor Peter Navarro told CNBC on Tuesday afternoon. The reversal came six hours after Trump announced his plan to double import duties on the Canadian metals in response to Ontario's decision to slap a 25% tax on electricity exports to the U.S. Ontario Premier Doug Ford said earlier Tuesday afternoon that he was pausing that surcharge following discussions w Trump’s Tariffs Set Off Day of Anger, Retaliation and Market Unease: the new new world\n\nAs the United States grapples with the upheaval unleashed by the Trump administration, many Chinese people are finding they can relate to what many Americans are going through.\n\nThey are saying it feels something like the Cultural Revolution, the period known as “the decade of turmoil.” The young aides Elon Musk has sent to dismantle the U.S. government reminded some Chinese of the Red Guards whom Mao Zedong enlisted to destroy the bureaucracy at the peak of the Cultural Revol",
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 7,276 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                                       | sentence_1                                                                                          | label                                                          |
  |:--------|:-------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                                           | string                                                                                              | float                                                          |
  | details | <ul><li>min: 65 characters</li><li>mean: 174.15 characters</li><li>max: 389 characters</li></ul> | <ul><li>min: 879 characters</li><li>mean: 1673.17 characters</li><li>max: 1834 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.42</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                            | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | label            |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>The D.C. National Guard was activated, and a citywide curfew was imposed. \| Trump supporters protested in Washington, D.C.</code>                                                              | <code>Pentagon faces deadline on recommending whether to invoke the Insurrection Act: Happy Friday! We are getting close to the 90-day deadline that President Donald Trump set back in January for the secretaries of the Departments of Defense and Homeland Security to recommend whether the president should invoke the Insurrection Act to address what he described as an “invasion” of gangs, human traffickers, and criminals at the southern border.<br><br>More than 10,000 service members are currently deployed to the U.S.-Mexico border. The Posse Comitatus Act of 1878 prevents federal troops Law enforcement missed key signs ahead of riot on US Capitol: CNN —<br><br>Despite weeks of planning between federal and local police agencies ahead of Wednesday’s Trump rally – including tracking social media – officials said that going into Wednesday they had no intelligence indicating there was a threat the US Capitol could be overrun.<br><br>It would turn out to be a catastrophic failure after an angry mob overwhelmed polic...</code>             | <code>1.0</code> |
  | <code>DeepSeek's app reached the top of U.S. free app charts within a week of its launch. \| DeepSeek unveiled its V3 model in December and R1 model in January.</code>                               | <code>DeepSeek R1 Surpasses ChatGPT and Google Gemini to Become the Top AI App Worldwide: Chinese technology company DeepSeek has made headlines globally with the launch of its open-source AI model, DeepSeek-R1. The model has rapidly gained popularity, igniting discussions across social media platforms and becoming a trending topic. Users have shared various memes and opinions, praising its capabilities and disruptive impact on the AI industry.<br><br>DeepSeek-R1’s associated application has surged in downloads, becoming the most downloaded free AI app on the US Apple App Store. It achieve Chinese tech start-up DeepSeek unnerves US with low-costs AI model on par with OpenAI's o1: deepseek<br><br>Chinese tech start-up DeepSeek is shaking the US' AI sector with its open-source approach and low-cost models, drawing curiosity and significant coverage from US media and AI forums about how this Chinese company has produced competitive AI systems comparable to those developed by US tech giants, all while navig...</code>                   | <code>1.0</code> |
  | <code>Widespread disruption occurred in internet and news access, and banks temporarily shut down across Myanmar following the coup. \| The election commission denied widespread voter fraud.</code> | <code>Myanmar’s military stages coup d’etat: Live news: In a series of morning raids, the military arrests senior government members and declares a state of emergency.<br><br>Myanmar’s military has seized power and declared a state of emergency for one year following days of escalating tension over the result of November’s parliamentary elections.<br><br>Aung San Suu Kyi, the country’s de facto leader, President Win Myint and other senior members of the National League for Democracy (NLD) party have been detained in the capital, Naypyidaw, on Monday.<br><br>A video br End of Myanmar’s Rocky Road to Democracy?: After days of speculation about an impending coup, Myanmar’s military has formally seized power the very day a newly elected parliament was scheduled to meet for the first time. Military generals ruled the country from the early 1960s until 2011. Now they are taking back control, after a near-decade of sharing power with elected lawmakers.<br><br>Political Veto by Coup<br><br>The coup began in the early hours of Fe...</code> | <code>0.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 8
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.5495 | 500  | 0.6151        |
| 1.0989 | 1000 | 0.2904        |
| 1.6484 | 1500 | 0.1988        |
| 2.1978 | 2000 | 0.1292        |
| 2.7473 | 2500 | 0.1195        |


### Framework Versions
- Python: 3.10.7
- Sentence Transformers: 5.2.0
- Transformers: 4.57.5
- PyTorch: 2.9.1
- Accelerate: 1.12.0
- Datasets: 4.4.2
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->