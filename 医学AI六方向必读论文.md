# 医学＋AI 六个方向的代表性必读论文

> 整理日期：2026-09-29。面向医学 AI 入门与课题选择，按用户提供的六方向截图组织。这里的“必读”是**代表性阅读路线**，不是按期刊影响因子排序，也不代表这些工作在不同数据集上可以直接比高低。每篇均给出原始论文或正式发表页面；“对我们启发”是研究设想，不是已验证的项目结论。

## 先看总图：六个方向分别解决什么

| 方向 | 核心对象与问题 | 典型研究方法 | 对当前选题的距离 |
| --- | --- | --- | --- |
| 1. 医学影像与计算机视觉 | 图像中的病灶在哪里、属于什么、能否跨医院工作 | 有监督学习、领域预训练、Transformer、提示分割 | **最近**：血细胞图像、病理切片、面部病灶均用得到 |
| 2. 临床 NLP 与电子病历 | 从报告、问诊和病历中提取或生成可靠信息 | 领域语言模型、任务微调、对话评测 | 若有合规报告/病历配对，才可能与影像结合 |
| 3. 基因组学与生物信息 | 从基因表达/序列推断细胞类型、调控或疾病机制 | 单细胞预训练、迁移学习、长序列建模 | 与**血细胞的分子机制**有关，但不等同于血细胞显微图像 |
| 4. 蛋白质、药物与 AI | 预测反应、结构及蛋白设计 | 序列建模、结构预测、生成模型 | 属于更远的生物医药路线，除非实验室有分子数据与验证合作 |
| 5. 多模态融合与临床决策支持 | 影像、文本、结构化指标怎样互相补充 | 图文对比学习、统一表征、跨模态注意力 | 有配对信息和明确临床终点时潜力大 |
| 6. 可解释性、公平性与临床验证 | 算法是否对真实人群有益、对谁会失效 | 偏倚审计、前瞻性试验、人机协作 | **所有方向必修**，决定论文的可信度而不只是分数 |

## 1. 医学影像与计算机视觉

**领域脉络：**先有大量标注图像上的专门模型，再有跨数据集预训练，进一步发展到可由点、框或文本提示的基础模型。这里特意保留分类、预训练和分割三类论文，避免把“检测/分类”误当成“精确分割”。

| 论文与正式来源 | 解决的问题 | 方法与主要发现 | 局限 / 对我们启发 |
| --- | --- | --- | --- |
| [CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison](https://ojs.aaai.org/index.php/AAAI/article/view/3834)，AAAI 2019 | 放射报告中的标签有“不确定”，如何构建可用的胸片分类数据集？ | 从报告构建不确定性标签，设计不同处理策略，并与专家比较。它展示了**标注定义会改变实验结果**。 | 是胸片**分类**，非病灶分割。血细胞/鲜红斑痣项目应先界定“不确定边界/疑似细胞”的标注规则，再比较网络。 |
| [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/abs/2102.04306)，预印本 2021 | U-Net 局部卷积特征在长距离关系上受限。 | CNN 提取局部纹理，Transformer 编码全局上下文，再由解码器恢复像素级结果。 | 加全局模块不保证跨设备稳健；在小数据集上需控制参数量并与 U-Net/nnU-Net 同划分比较。此处引用的是**2021 预印本**，不混同于后来题名不同的期刊版本。 |
| [Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images](https://arxiv.org/abs/2201.01266)，预印本 2022 | 3D 医学体数据如何兼顾多尺度与大范围上下文？ | 用分层 Swin Transformer 编码 3D 体块，配合 U 形解码器进行脑肿瘤分割。 | 研究对象是 **MRI 体数据**，不能把 2D 全切片称作 3D 病理。若实验室确有连续切片，才考虑其 3D 设计；显存成本也须评估。 |
| [Segment Anything in Medical Images（MedSAM）](https://www.nature.com/articles/s41467-024-44824-z)，Nature Communications 2024 | 通用 SAM 直接用于医学图像时，域差异和提示方式会限制效果。 | 汇集多种医学模态分割数据对 SAM 做领域适配，提供提示式医学分割基线。 | “能提示分割”≠“无人提示自动临床可用”。面部病灶应比较固定提示、自动提示、无提示设置，并记录提示成本。 |
| [RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC9530758/)，Radiology: Artificial Intelligence 2022 | 自然图像 ImageNet 权重是否是医学影像迁移的最佳起点？ | 建立放射影像预训练资源；论文报告多个下游任务上较 ImageNet 预训练有优势。 | CT/MRI/超声的特征并不必然适合血涂片或皮肤照片。迁移实验应设随机初始化、ImageNet、医学预训练三个对照，并按同一训练预算比较。 |

**这一方向尚未解决的关键问题：**跨医院/设备的域偏移、标注者分歧、少标注样本、边界不清与真实工作流成本。单一数据集的 Dice 高，并不能替代外部验证和失败样本分析。

## 2. 临床 NLP 与电子病历

**领域脉络：**从专业术语预训练、临床笔记适配，走向大规模电子病历模型和对话式诊疗。语句流畅与医学正确不是同一回事。

| 论文与正式来源 | 解决的问题 | 方法与主要发现 | 局限 / 对我们启发 |
| --- | --- | --- | --- |
| [BioBERT: a pre-trained biomedical language representation model for biomedical text mining](https://doi.org/10.1093/bioinformatics/btz682)，Bioinformatics 2020 | 通用 BERT 对生物医学术语理解不足。 | 在生物医学语料继续预训练并微调实体识别、关系抽取等任务，证明领域文本预训练的价值。 | 主要处理文献文本，不是直接的临床决策器。若后续整理病理报告，先把实体和诊断标签定义清楚。 |
| [ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission](https://arxiv.org/abs/1904.05342)，预印本 2019 / CHIL workshop 2020 | 临床笔记的用词、缩写和结构不同于科研文献。 | 在临床记录上适配语言模型，用于再入院预测等任务。 | 数据场景偏特定医院及任务，且该条不能写成“2020 顶刊论文”。启发是**报告语料与论文语料应分开评估**。 |
| [A large language model for electronic health records（GatorTron）](https://www.nature.com/articles/s41746-022-00742-2)，npj Digital Medicine 2022 | 电子病历中复杂语义关系难以由小型临床语言模型充分表示。 | 大规模临床文本预训练，并在多项临床 NLP 任务上评估。 | 受数据权限、群体构成及部署成本约束。对我们的意义是“真实院内数据治理”先于模型扩容。 |
| [Large language models encode clinical knowledge（Med-PaLM）](https://www.nature.com/articles/s41586-023-06291-2)，Nature 2023 | 大语言模型的医学回答是否既有知识又安全？ | 建立医学问答评测并对模型回答作人工医学质量评价。 | 问答基准不等于真实诊疗；可能出现事实错误或伤害性建议。若用语言模型辅助标注/报告，只能作为待核验工具。 |
| [Towards conversational diagnostic artificial intelligence（AMIE）](https://www.nature.com/articles/s41586-025-08866-7)，Nature 2025 | 对话问诊能否更系统地收集病史并形成鉴别诊断？ | 构建诊断对话模型，在模拟文本问诊中进行评估。 | 模拟、文本环境不能直接推断门诊疗效或患者安全性。若做临床多模态研究，须设计真实场景、医生参与和风险控制。 |

**争议焦点：**自动指标、专家盲评、真实患者结局分别回答不同问题；临床报告可能包含隐私，不能因为模型需要语料就把院内文本上传外部服务。

## 3. 基因组学与生物信息

**领域脉络：**把单细胞表达谱或 DNA 序列转换为机器可学习的表示，再检验能否迁移到未见细胞类型或生物学任务。注意：这里的“细胞”多指**分子测量样本**，不是显微照片里的细胞实例。

| 论文与正式来源 | 解决的问题 | 方法与主要发现 | 局限 / 对我们启发 |
| --- | --- | --- | --- |
| [scBERT as a large-scale pretrained deep language model for cell type annotation of single-cell RNA-seq data](https://www.nature.com/articles/s42256-022-00534-z)，Nature Machine Intelligence 2022 | 少标注条件下如何做单细胞 RNA 测序的细胞类型注释？ | 在大量未标注表达谱上预训练，再微调注释任务。 | [2024 年独立复核](https://www.nature.com/articles/s42256-024-00949-w)发现，在其测试的两个数据集上简单逻辑回归可相当或更好，且移除预训练影响有限；原作者有[回应](https://www.nature.com/articles/s42256-024-00948-x)。启发：**复杂模型必须赢过强而简单的基线，并公开消融**。 |
| [Transfer learning enables predictions in network biology（Geneformer）](https://www.nature.com/articles/s41586-023-06139-9)，Nature 2023 | 小样本条件下如何预测基因网络相关状态与扰动效应？ | 在约三千万单细胞转录组上预训练，再迁移到网络生物学任务。 | 表达谱批次差异和训练语料重叠可能影响外推；血细胞**图像**项目不能直接宣称使用 Geneformer 就解决形态识别。 |
| [Effective gene expression prediction from sequence by integrating long-range interactions（Enformer）](https://www.nature.com/articles/s41592-021-01252-x)，Nature Methods 2021 | 只看短片段 DNA 会漏掉远距离调控关系。 | 长序列模型综合远程作用来预测基因表达相关信号。 | 预测调控信号不是直接证明因果机制；从血液图像延伸到基因机制，需要有真实的配对分子数据。 |

**争议焦点：**预训练规模是否真的带来迁移收益？是否按供体、实验批次和中心隔离训练/测试？如果“同一供体的不同细胞”同时进两组，结果可能虚高。

## 4. 蛋白质、药物与 AI

**领域脉络：**先把化学反应当作序列转换，再预测蛋白质结构与复合体，最后尝试按目标修改现有蛋白。它们是重要方法背景，但并非当前血细胞显微图像任务的直接基线。

| 论文与正式来源 | 解决的问题 | 方法与主要发现 | 局限 / 对我们启发 |
| --- | --- | --- | --- |
| [Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction](https://doi.org/10.1021/acscentsci.9b00576)，ACS Central Science 2019 | 已知反应物，如何预测产物并估计模型不确定性？ | 以 SMILES 序列和 Transformer 预测反应产物，并分析置信度。 | 专利反应数据与真实合成可行性不同；启发是**不只输出预测，还要让模型表达不确定性**。 |
| [Highly accurate protein structure prediction with AlphaFold](https://www.nature.com/articles/s41586-021-03819-2)，Nature 2021（AlphaFold 2） | 从氨基酸序列预测蛋白结构的准确度长期受限。 | 结合进化信息与端到端结构建模，显著推进单体结构预测。 | 高结构可信度不自动证明动力学或功能；这篇应归入**蛋白质结构**，不是基因组学。 |
| [Accurate structure prediction of biomolecular interactions with AlphaFold 3](https://www.nature.com/articles/s41586-024-07487-w)，Nature 2024 | 从蛋白单体走向多种生物分子相互作用结构。 | 扩展到蛋白、核酸、配体等复合物的结构建模。 | 预测结合构象不等同于体内药效和安全性；如做药物方向，必须设计湿实验或可信外部验证。 |
| [Evolutionary-scale prediction of atomic-level protein structure with a language model](https://doi.org/10.1126/science.ade2574)，Science 2023（ESMFold） | 是否可以从蛋白序列语言表示快速推断结构？ | 蛋白语言模型驱动结构预测，凸显大规模自监督序列学习的潜力。 | 与 AlphaFold 的输入和速度—精度权衡不同；截图中的“ESM-2＝2023 Science 结构预测论文”不精确，**ESM-2 是模型家族，ESMFold 是结构预测工作**。 |
| [Miniaturizing and modifying natural proteins with Raygun](https://www.nature.com/articles/s41586-026-10842-8)，Nature 2026 | 如何以天然蛋白为模板，同时控制替换、插入、删除并尽量保留功能？ | 用定长概率表示生成不同长度的蛋白变体，报告部分细胞实验验证。 | 已测试蛋白和功能仍有限，不能概括为所有蛋白均可任意改造；与我们当前影像课题联系较远。截图给出的“Parameterized redesign of proteins”不是该正式题名。 |

**争议焦点：**结构预测、功能预测、可合成性与临床可用性是不同终点，不能用一个结构分数替代后续实验证据。

## 5. 多模态融合与临床决策支持

**领域脉络：**单幅图像可能无法回答临床问题；加入主诉、报告或化验后，模型可能更有用。但如果配对、时间顺序或缺失情况不清楚，融合也会制造虚假的提升。

| 论文与正式来源 | 解决的问题 | 方法与主要发现 | 局限 / 对我们启发 |
| --- | --- | --- | --- |
| [Towards Generalist Biomedical AI（Med-PaLM M）](https://arxiv.org/abs/2307.14334)，预印本 2023 / NEJM AI 2024 | 一个模型能否处理医学文本、影像等多任务输入？ | 建立多模态通用生物医学模型与多任务评测，展示跨任务能力。 | 通用能力不表示在每种细分病灶上优于专用模型；若实验室只有图像，不宜虚构多模态任务。 |
| [A transformer-based representation-learning model with unified processing of multimodal input for clinical diagnostics（IRENE）](https://www.nature.com/articles/s41551-023-01045-x)，Nature Biomedical Engineering 2023 | 影像、主诉、结构化临床信息怎样在诊断时互补？ | 用统一 Transformer 和跨模态注意力融合输入，在所研究的临床诊断任务中优于若干非统一融合基线。 | 回顾性任务的可用性依赖数据完整、采集时间正确；若有血细胞形态＋血常规，应防止“检查结果发生在预测时点之后”的信息泄漏。 |
| [Expert-level detection of pathologies from unannotated chest X-ray images via self-supervised learning（CheXzero）](https://www.nature.com/articles/s41551-022-00936-9)，Nature Biomedical Engineering 2022 | 没有逐类人工标签时，能否利用影像及报告做疾病识别？ | 图文对比学习让胸片与报告对齐，在论文设置中支持零样本病变**分类/检测**。 | 不是像素级病灶分割；报告可能含诊断捷径。可借鉴弱监督思路，但必须用独立标注验证定位。 |

**争议焦点：**“融合后更高分”是否来自真正互补，还是来自一模态直接泄漏答案？至少比较图像单独、文本单独、融合、缺失模态四种情况，并说明每条数据的时间顺序。

## 6. 可解释性、公平性与临床验证

**领域脉络：**准确率只是实验室阶段的起点。真正的问题是：换医院、设备和人群之后还能用吗？医生使用后会不会改善决策？以下综述、偏倚审计和随机试验形成三级证据。

| 论文与正式来源 | 解决的问题 | 方法与主要发现 | 局限 / 对我们启发 |
| --- | --- | --- | --- |
| [High-performance medicine: the convergence of human and artificial intelligence](https://www.nature.com/articles/s41591-018-0300-7)，Nature Medicine 2019，综述 | 医学 AI 如何与医生的能力和工作流结合？ | 讨论影像判读、流程效率，以及偏倚、隐私、透明性风险。 | 是**综述/观点整合**，不是某个模型临床有效性的试验。读后要把“人机协作怎么测”写入实验方案。 |
| [AI in health and medicine](https://www.nature.com/articles/s41591-021-01614-0)，Nature Medicine 2022，综述 | 医学 AI 从回顾性算法走向临床使用，还缺哪些证据？ | 归纳前瞻性研究、影像分析和人机协作方向。 | 截图中的“AI for health: State of the art, challenges, future directions，Rajpurkar，NEJM AI 2024”未找到对应的准确书目信息；此处改列**可核实的真实论文**。 |
| [Dissecting racial bias in an algorithm used to manage the health of populations](https://doi.org/10.1126/science.aax2342)，Science 2019 | 高总体准确率的医疗算法为何仍可能系统性不公平？ | 审计真实医疗资源配置算法，发现以医疗花费作健康需求代理会带来种族偏倚。 | 研究对象不是分割模型；可迁移的是**检查标签、代理变量及各亚组错误率**。若没有合法群体属性，不能臆造公平性结论。 |
| [MASAI trial: AI-supported screen reading versus standard double reading](https://pubmed.ncbi.nlm.nih.gov/37541274/)，The Lancet Oncology 2023 | AI 参与乳腺筛查能否在保证早期安全性的同时减少读片工作量？ | 随机对照临床研究的预设早期安全性分析，报告 AI 辅助组阅片工作量下降，癌症检出率达到预设安全下限。 | **该篇不是长期终点的最终结论**；不能据此宣称所有 AI 筛查都安全。启发是预先定义终点、临床工作量和随访，而不只报模型 AUC。 |

**争议焦点：**热力图不等于机制解释；分数高不等于公平；回顾性内部测试不等于前瞻性临床获益。研究设计要把这三件事分开证明。

## 对我们课题的综合判断

1. **最近的可执行研究链：医学影像基础模型/分割 → 严格数据划分 → 失败分析 → 外部或跨域验证。**血细胞可先做检测/分类/开放集，面部鲜红斑痣可做分割与纵向一致性，但必须分别定义临床任务；不要仅把新网络加到 U-Net 上当创新。
2. **三维病理要先确认数据形态。**如果只有一张张 2D 全切片图像，题目应称数字病理/WSI；只有连续切片、空间间距与可验证对应关系齐备，才能研究组织 3D 重建或 3D 分割。
3. **多模态创新建立在配对与时间正确之上。**真正有血常规、血涂片和患者级结局的配对数据，才值得做图像＋结构化指标融合；没有这些条件，先做高质量单模态基线更诚实。
4. **最值得借鉴的实验习惯**是：简单强基线（含不使用预训练）、患者/供体级隔离、不同设备或中心测试、亚组分析、消融、失败样本归因以及标注不确定性复核。scBERT 后续争论提醒我们，模型复杂度不是贡献本身。
5. **截图未覆盖的空白**：血细胞显微图像的真实开放集类别和标注协议、WSI 与真正 3D 病理的区别、面部鲜红斑痣的拍摄标准化/颜色量化/随访终点。若最后选定其中一题，还需补专门的近年综述和直接竞争方法；本清单只负责建立共同方法论。

## 建议阅读顺序与读法

**第一轮，先读与论文实验最近的 6 篇：**CheXpert（标签）、TransUNet（架构）、RadImageNet（预训练）、MedSAM（领域模型）、scBERT 及其独立复核（强基线与可复现性）、MASAI（真实临床证据）。不要求一次理解所有公式，先为每篇写下“任务—数据—划分—对照—指标—最可能的失败原因”。

**第二轮，根据导师最终选题分叉：**选血细胞图像，补血细胞专门检测/分割与开放集论文；选三维病理，补连续切片配准和 3D 重建；选鲜红斑痣，补临床照片标准化、颜色量化及疗效评价。基因组、蛋白和 NLP 方向可以了解思想，但不用在缺少对应数据时急着复现。

**给导师汇报可用的一句话：**“我先用六方向代表作建立医学 AI 的方法地图，初步判断最接近我们现有能力的是医学影像任务；下一步想先请您明确血细胞/三维病理/鲜红斑痣中哪一个有合规数据、标注和临床合作，然后围绕该任务建立强基线与失败分析，避免只追逐模型名称。”

## 书目信息与编写边界

- 年份采用正式发表年份；标为“预印本”的条目不伪装为同期正式期刊论文。截图中的误写已在相应条目纠正，包括 GatorTron 正式题名、Enformer 期刊、AlphaFold 2 分类、ESM-2/ESMFold 区分、Raygun 题名和 Rajpurkar 综述。
- 表中的“发现”限于各论文报告的任务和评价设置，**不同论文的成绩不可横比**；“局限/启发”部分包含基于研究设计的推论，需以后续专门实验验证。
- 本文是公开文献的教学性梳理，不使用、不展示任何患者图像、身份映射或私有标注。编写时参考了 [Kassis 等人的科研写作证据溯源与核查建议](https://arxiv.org/abs/2609.00065)，并对照上述原始论文页面核对题名、载体和主要主张。
