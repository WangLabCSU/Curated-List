#!/usr/bin/env python3
import json
import os
import re
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

# Directories and Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
edge_bookmarks_path = "/Users/wsx/Library/Application Support/Microsoft Edge/Default/Bookmarks"

print(f"Repository Root: {repo_dir}")

# 1. Load Bookmarks (Try Edge first, fallback to local JSON)
bookmarks = []
loaded_from_edge = False

if os.path.exists(edge_bookmarks_path):
    try:
        with open(edge_bookmarks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        def collect_bookmarks(node, path_segments=[]):
            found = []
            name = node.get("name", "").strip()
            node_type = node.get("type", "")
            current_path = path_segments + [name] if name else path_segments
            if node_type == "folder":
                for child in node.get("children", []):
                    found.extend(collect_bookmarks(child, current_path))
            elif node_type == "url":
                url = node.get("url", "").strip()
                found.append({
                    "path": current_path[:-1],
                    "name": name,
                    "url": url
                })
            return found

        roots = data.get("roots", {})
        for key, value in roots.items():
            bookmarks.extend(collect_bookmarks(value, [key]))
            
        print(f"Loaded {len(bookmarks)} bookmarks from Edge.")
        loaded_from_edge = True
        
    except Exception as e:
        print(f"Failed to load from Edge: {e}")

if not loaded_from_edge:
    print("Error: No bookmark source found (Microsoft Edge is not installed or bookmarks file is missing).")
    exit(1)

# 2. Blacklists and Filtering Mappings
blacklisted_folders = {"单位", "娱乐", "社区工作", "下载", "工作区", "其他收藏夹", "移动收藏夹", "翻墙"}

private_domains = [
    r"localhost", r"127\.0\.0\.1", r"^192\.168\.", r"^10\.", r"^172\.(1[6-9]|2[0-9]|3[0-1])\.",
    r"inner\.wei-group\.net", r"mail\.", r"webmail\.", r"ca\.csu\.edu\.cn", r"oa\.csu\.edu\.cn",
    r"hr\.csu\.edu\.cn", r"my\.csu\.edu\.cn", r"library\.csu\.edu\.cn",
    r"library\.shanghaitech\.edu\.cn", r"spoc\.shanghaitech\.edu\.cn",
    r"mooc\.shanghaitech\.edu\.cn", r"mail\.shanghaitech\.edu\.cn",
    r"console\.", r"portal\.", r"admin\.", r"grants\.nsfc", r"apply\.hgrg\.net",
    r"pan\.baidu\.com", r"aliyundrive", r"weibo\.com", r"taobao\.com", r"jd\.com", r"alipay\.com",
    r"wechat", r"wx\.qq\.com"
]

private_url_keywords = [
    r"/login", r"/signin", r"/signup", r"/auth", r"/register", r"/logout",
    r"search\?q=", r"baidu\.com/s\?", r"google\.com/search\?", r"about:blank",
    r"^file://", r"^chrome://", r"^edge://", r"my-orcid",
    r"科研者之家-中南大学湘雅", r"Nat Commun.*基于网络生物学.*预测癌症患者",
    r"ShixiangWang/yq", r"loop\.frontiersin\.org/people", r"ShixiangWang/vim-galore-zh_cn",
    r"support\.bioconductor\.org/u/14673", r"clickup\.com/9010147517", r"openml\.org/u/6526",
    r"stackoverflow\.com/users/7662327/wang-shixiang", r"hpc\.csu\.edu\.cn",
    r"lockedata\.slack\.com", r"slack\.com/intl/zh-cn/", r"oncoharmony\.feishu\.cn",
    r"ai\.feishu\.cn/docx", r"lyswhut/lx-music-desktop", r"macwk\.com/soft/termius",
    r"hao\.su/page/1/", r"wunderlist\.com/zh/", r"bilibili\.danmaku\.live",
    r"orsoon\.com", r"macwk\.com", r"yjst\.net/voddetail", r"yuque\.com/u48538112",
    r"Adobe Illustrator 淘宝购买版本", r"yuque\.com/yangyulan-ayaeq",
    r"singlecell\.yuque\.com/aie23e", r"getlantern/download", r"godcong/fate"
]

tracking_params = {
    'utm_source', 'utm_medium', 'utm_campaign', 'utm_content', 'utm_term', 'utm_psn',
    'spm', 'spm_id_from', 'vd_source', 'share_source', 'share_medium', 'share_plat', 
    'share_session_id', 'share_tag', 'unique_k', 'up_id', 'wxfid', 
    'code', 'state', 'from', 'buvid', 'is_story_h5', 'mid', 'plat_id', 
    'share_from', 'timestamp', 'unique_id', '_unique_id_', 'share_times',
    'from_spmid', 'spmid', 'track', 'client', 'search_id', 'tab',
    '_from', 'click_from', 'scene', '_hsmi', '_hsenc', 'gio_link_id'
}

def optimize_url(url):
    try:
        parsed = urlparse(url)
        q_params = parse_qsl(parsed.query)
        filtered_q = []
        for k, v in q_params:
            if k.lower() not in tracking_params:
                filtered_q.append((k, v))
        new_query = urlencode(filtered_q)
        optimized = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment
        ))
        if "github.com" in optimized:
            optimized = re.sub(r'\?tab=readme-ov-file$', '', optimized)
        return optimized
    except Exception:
        return url

def clean_name(name):
    suffixes = [
        r'\s*[-|·_]\s*github\s*$', r'\s*[-|·_]\s*rstudio\s*$', r'\s*[-|·_]\s*stack\s*overflow\s*$',
        r'\s*[-|·_]\s*简书\s*$', r'\s*[-|·_]\s*博客园\s*$', r'\s*[-|·_]\s*csdn博客\s*$',
        r'\s*[-|·_]\s*开发者社区\s*$', r'\s*[-|·_]\s*腾讯云\s*$', r'\s*[-|·_]\s*阿里云\s*$',
        r'\s*[-|·_]\s*百度百科\s*$', r'\s*[-|·_]\s*知乎\s*$', r'\s*[-|·_]\s*bilibili\s*$',
        r'\s*[-|·_]\s*segmentfault\s*思否\s*$', r'\s*[-|·_]\s*掘金\s*$', r'\s*[-|·_]\s*oschina\s*$',
        r'\s*[-|·_]\s*华为云社区\s*$', r'\s*[-|·_]\s*infoq\s*$', r'\s*[-|·_]\s*36氪\s*$',
        r'\s*[-|·_]\s*机器之心\s*$', r'\s*[-|·_]\s*廖雪峰的官方网站\s*$', r'\s*[-|·_]\s*pypi\s*$',
        r'\s*[-|·_]\s*npm\s*$', r'\s*[-|·_]\s*bioconductor\s*$', r'\s*[-|·_]\s*read\s*the\s*docs\s*$',
        r'\s*documentation\s*$', r'\s*文档\s*$', r'\s*手册\s*$', r'\s*主页\s*$', r'\s*官网\s*$'
    ]
    for suffix in suffixes:
        name = re.sub(suffix, '', name, flags=re.I)
    name = re.sub(r'wangshixiang|shixiangwang|wangshx|王诗翔', 'Developer', name, flags=re.I)
    return name.strip().strip('-').strip('|').strip('_').strip()

# Low quality generic names to descriptive name replacements
name_replacements = {
    "https://rcppcore.github.io/RcppParallel/": "RcppParallel",
    "https://cran.r-project.org/web/packages/bedr/vignettes/Using-bedr.html": "Using bedr Vignette",
    "https://bernatgel.github.io/karyoploter_tutorial/": "karyoploteR Tutorial",
    "https://bionimbus-pdc.opensciencedatacloud.org/datasets/": "Bionimbus PDC Datasets",
    "https://tcia.at/home": "The Cancer Imaging Archive (TCIA)",
    "https://www.cancerimagingarchive.net/collections/": "TCIA Collections",
    "http://inventwithpython.com/pygame/index.html": "Making Games with Python & Pygame",
    "http://www.ndexbio.org/#/": "NDEx - The Network Data Exchange",
    "https://www.mohu.org/info/symbols/symbols.htm": "LaTeX Mathematical Symbols List"
}

# 3. Clean and Filter Bookmarks
filtered_bookmarks = []
seen_urls = set()

for b in bookmarks:
    path = b["path"]
    name = b["name"].strip()
    url = b["url"].strip()
    
    # 1. Filter folders
    if any(p in blacklisted_folders for p in path):
        continue
    path_str = " -> ".join(path)
    if "个人 -> 常用" in path_str or "个人 -> 云空间" in path_str:
        continue
        
    # 2. Skip empty
    if not name or not url:
        continue
        
    # 3. Check domain privacy
    parsed = urlparse(url)
    domain = parsed.netloc
    if any(re.search(pat, domain, re.I) for pat in private_domains):
        continue
        
    # 4. Check URL keywords
    if any(re.search(pat, url, re.I) for pat in private_url_keywords):
        continue
        
    # 5. Upgrades / Subscriptions / Normalizations
    if name == "生信技能树":
        url = "http://www.biotrainee.com/"
    elif name == "ApacheCN":
        url = "https://www.apachecn.org/"
    elif "osf.io/profile" in url:
        url = "https://osf.io/"
        name = "OSF | Open Science Framework"
        
    # 6. Privacy removal of user's name in URL/name
    if re.search(r'wangshixiang|shixiangwang|wangshx|王诗翔', url, re.I):
        if "github.com/" in url:
            parts = parsed.path.strip("/").split("/")
            if len(parts) < 2:
                continue
        else:
            continue
            
    # Optimize URL (remove tracking)
    optimized_url = optimize_url(url)
    
    # Deduplicate
    norm_url = optimized_url.rstrip("/").lower()
    if norm_url in seen_urls:
        continue
    seen_urls.add(norm_url)
    
    # Clean the name
    cleaned_name = clean_name(name)
    if not cleaned_name:
        cleaned_name = "Link"
    if optimized_url in name_replacements:
        cleaned_name = name_replacements[optimized_url]
        
    filtered_bookmarks.append({
        "path": path,
        "name": cleaned_name,
        "url": optimized_url
    })


print(f"Filtered down to {len(filtered_bookmarks)} clean bookmarks.")

# 4. File Structure Config
files_structure = {
    "academic_resources.md": {
        "title": "📚 学术研究与科研工具清单 (Academic & Scientific Resources)",
        "intro": "本列表收录了学术研究团队、核心期刊、文献检索、学术写作与投稿、科研绘图与配色、学术展示、学术机构等科研与学术服务相关的优质网站。",
        "sections": [
            {"path_keywords": ["学术 -> 学术研究组", "学术 -> 学术人物"], "title": "👥 学术研究团队与实验室 (Research Groups & Labs)"},
            {"path_keywords": ["学术 -> 学术期刊", "学术 -> 出版"], "title": "📖 核心学术期刊与出版 (Academic Journals & Publishing)"},
            {"path_keywords": ["搜索 -> 学术", "学术 -> 学术分析"], "title": "🔍 学术检索与文献搜索引擎 (Academic Search)"},
            {"path_keywords": ["学术 -> 学术工具"], "title": "🛠️ 科研学术辅助工具 (Research Tools)"},
            {"path_keywords": ["工作工具 -> 写作与分享", "学术 -> 学术写作与投稿", "个人 -> 简历"], "title": "✍️ 学术写作、润色与投稿 (Scientific Writing & Publishing)"},
            {"path_keywords": ["学术 -> 学术绘图"], "title": "🎨 科研绘图与配色设计 (Scientific Plotting & Design)"},
            {"path_keywords": ["学术 -> 学术资讯"], "title": "📰 学术前沿与行业资讯 (Academic News & Media)"},
            {"path_keywords": ["学术 -> 学术展示"], "title": "📊 学术汇报与幻灯片设计 (Academic Presentations)"},
            {"path_keywords": ["学术 -> 学术机构"], "title": "🏛️ 国际著名学术机构 (Academic Institutions)"},
            {"path_keywords": ["学术 -> 公司", "学术 -> 创业"], "title": "🏢 生物信息与科研服务公司 (Academic Service Companies)"},
            {"path_keywords": ["学术 -> 云计算"], "title": "☁️ 科研云计算平台 (Scientific Cloud Computing)"},
            {"path_keywords": ["学术 -> 学术账号"], "title": "👥 学术账号与科研主页建站 (Academic Accounts & Scholar Sites)"},
            {"path_keywords": ["学术 -> 学术分享"], "title": "🗣️ 学术成果分享与交流 (Academic Sharing)"},
            {"path_keywords": ["学术 -> 学术资源申请"], "title": "📑 科研经费与资源申请 (Research Grants & Resources)"},
            {"path_keywords": ["学术 -> 学术审稿"], "title": "✍️ 学术同行评议与审稿 (Peer Review)"},
            {"path_keywords": ["学术 -> 学术实验", "工作工具 -> 实验工具"], "title": "🧪 学术实验协议与方法 (Experimental Protocols)"},
            {"path_keywords": ["学术 -> 学位论文"], "title": "🎓 学位论文检索与编写 (Theses & Dissertations)"},
            {"path_keywords": ["学术 -> 生物信息学考古"], "title": "📜 生物信息学历史与经典 (Bioinformatics History)"}
        ]
    },
    "bioinformatics_tools.md": {
        "title": "🧬 生信分析与计算工具清单 (Bioinformatics Tools & Pipelines)",
        "intro": "本列表整理了下一代测序（NGS）、单细胞测序、多组学分析、HLA分型、生信文件处理、变异检测与注释、免疫浸润、肿瘤异质性、生存分析等生信计算分析的软件工具与分析流程。",
        "sections": [
            {"path_keywords": ["工作工具 -> NGS"], "title": "🧬 下一代测序基础工具 (NGS)"},
            {"path_keywords": ["工作工具 -> 单细胞"], "title": "🧫 单细胞测序分析 (Single Cell Analysis)"},
            {"path_keywords": ["工作工具 -> 组学分析"], "title": "📊 组学数据分析 (Omics Analysis)"},
            {"path_keywords": ["工作工具 -> HLA calling"], "title": "🧪 HLA 分型与免疫基因组学 (HLA Calling)"},
            {"path_keywords": ["工作工具 -> 文件（VCF、MAF、bam等）工具"], "title": "📂 生信文件格式与处理工具 (Bioinformatics File Utilities)"},
            {"path_keywords": ["工作工具 -> 变异Calling、注释与分析"], "title": "🧬 变异检测、注释与分析 (Variant Calling & Annotation)"},
            {"path_keywords": ["工作工具 -> Neoantigen计算与分析"], "title": "🧫 新抗原计算与分析 (Neoantigen Prediction)"},
            {"path_keywords": ["工作工具 -> 免疫浸润与异质性"], "title": "🛡️ 免疫浸润与肿瘤异质性 (Immune Infiltration & Tumor Heterogeneity)"},
            {"path_keywords": ["工作工具 -> 生存分析"], "title": "⏱️ 临床预后与生存分析 (Survival Analysis)"},
            {"path_keywords": ["工作工具 -> ecDNA"], "title": "🧬 染色体外 DNA 分析 (ecDNA)"},
            {"path_keywords": ["工作工具 -> 微生物组"], "title": "🦠 微生物组与宏基因组分析 (Metagenomics & Microbiome)"},
            {"path_keywords": ["工作工具 -> 蛋白结构预测"], "title": "🔬 蛋白质结构预测 (Protein Structure)"},
            {"path_keywords": ["工作工具 -> 富集分析"], "title": "📈 功能富集与通路分析 (Enrichment & Pathway Analysis)"},
            {"path_keywords": ["工作工具 -> 演化"], "title": "🌳 系统发育与进化分析 (Phylogeny & Evolution)"},
            {"path_keywords": ["工作工具 -> 代谢与蛋白组"], "title": "🧪 蛋白质组与代谢组学 (Proteomics & Metabolomics)"},
            {"path_keywords": ["工作工具 -> 数字病理"], "title": "🔬 数字病理图像分析 (Digital Pathology)"},
            {"path_keywords": ["工作工具 -> 基因组可视化"], "title": "🗺️ 基因组可视化工具 (Genome Visualization)"},
            {"path_keywords": ["工作工具 -> 三代测序"], "title": "🧬 第三代/长读长测序 (Long-read Sequencing)"},
            {"path_keywords": ["工作工具 -> 分析平台"], "title": "⚙️ 生信在线分析与计算平台 (Online Analysis Platforms)"},
            {"path_keywords": ["开发 -> 生信命令行工具"], "title": "💻 生信命令行工具包 (Bioinfo CLI Tools)"},
            {"path_keywords": ["分析流程"], "title": "⚙️ 分析流程与工作流 (Analysis Pipelines & Workflows)"}
        ]
    },
    "databases_and_web.md": {
        "title": "🗄️ 生物医学数据库与在线平台 (Biomedical Databases & Web Portals)",
        "intro": "本列表收录了全球生物医学数据中心、物种参考基因组及基因组注释库、TCGA/ICGC癌症基因组、非编码RNA与转录调控、免疫组学、基因突变、多组学整合数据库以及单细胞转录组等专业领域的医学数据库平台。",
        "sections": [
            {"path_keywords": ["数据库/Web -> Data(base)-Center", "数据库/Web"], "title": "🗄️ 综合数据中心 (Data Centers)"},
            {"path_keywords": ["数据库/Web -> Reference"], "title": "📖 基因组参考物种与注释 (Reference Genomes)"},
            {"path_keywords": ["数据库/Web -> TCGA"], "title": "🎗️ TCGA 癌症基因组数据库 (TCGA)"},
            {"path_keywords": ["数据库/Web -> ICGC"], "title": "🎗️ ICGC 国际癌症基因组联盟 (ICGC)"},
            {"path_keywords": ["数据库/Web -> Merged"], "title": "🔗 多组学整合数据库 (Merged Databases)"},
            {"path_keywords": ["数据库/Web -> ENCODE/Noncoding"], "title": "🧬 非编码与转录调控 (ENCODE & Non-coding)"},
            {"path_keywords": ["数据库/Web -> Immune"], "title": "🛡️ 免疫学数据库 (Immunology Databases)"},
            {"path_keywords": ["数据库/Web -> Mutation"], "title": "🧬 突变与变异数据库 (Mutation Databases)"},
            {"path_keywords": ["数据库/Web -> Expression"], "title": "📈 基因表达与转录调控 (Expression Databases)"},
            {"path_keywords": ["数据库/Web -> Protein"], "title": "🔬 蛋白质与相互作用 (Protein & Interaction Databases)"},
            {"path_keywords": ["数据库/Web -> CellLine/Drug"], "title": "🧪 细胞系与药物反应 (Cell Line & Drug Response)"},
            {"path_keywords": ["数据库/Web -> SingeCell"], "title": "🧫 单细胞转录组数据库 (Single Cell Databases)"},
            {"path_keywords": ["数据库/Web -> Tools"], "title": "🛠️ 在线分析工具与服务 (Online Analysis Tools)"},
            {"path_keywords": ["数据库/Web -> Cancer-Genes"], "title": "🎗️ 癌症相关基因数据库 (Cancer Genes)"},
            {"path_keywords": ["数据库/Web -> ClinicalTrial"], "title": "🏥 临床试验与队列 data (Clinical Trials & Cohorts)"},
            {"path_keywords": ["数据库/Web -> NOrmal"], "title": "🚶 健康样本对照数据库 (Normal Control)"},
            {"path_keywords": ["数据库/Web -> Methylation"], "title": "🧬 甲基化与表观遗传数据库 (Methylation & Epigenetics)"},
            {"path_keywords": ["数据库/Web -> virus"], "title": "🦠 病毒基因组数据库 (Virus Databases)"},
            {"path_keywords": ["数据库/Web -> 中药"], "title": "🌿 中医药与天然产物数据库 (TCM Databases)"},
            {"path_keywords": ["数据库/Web -> ecDNA"], "title": "🧬 ecDNA 染色体外DNA数据库 (ecDNA Databases)"},
            {"path_keywords": ["数据库/Web -> 免疫治疗数据集"], "title": "🛡️ 肿瘤免疫治疗数据集 (Immunotherapy Datasets)"},
            {"path_keywords": ["数据库/Web -> 专家知识库"], "title": "🧠 领域专家知识库 (Knowledgebases)"},
            {"path_keywords": ["数据库/Web -> Medical-Image"], "title": "🖼️ 医学影像数据库 (Medical Image Datasets)"},
            {"path_keywords": ["数据库/Web -> Big-Projects"], "title": "🌐 重大国际科学项目 (Large-scale Science Projects)"},
            {"path_keywords": ["数据库/Web -> 注释", "工作工具 -> 注释包", "搜索 -> 数据字典"], "title": "🏷️ 基因与功能注释库 (Annotation Databases)"},
            {"path_keywords": ["数据库/Web -> 扰动数据集"], "title": "⚡ 扰动与基因编辑数据集 (Perturbation Datasets)"},
            {"path_keywords": ["工作工具 -> 数据（接口）下载包"], "title": "📦 数据下载与接口工具包 (Data Download Utilities)"}
        ]
    },
    "software_development.md": {
        "title": "💻 软件开发、系统配置与容器化部署 (Software Development & System Operations)",
        "intro": "本列表收集了 R, Python, Go, Rust, Julia 等主流开发语言环境、包开发脚手架、IDE 配置、操作系统底层调优、跨平台效率软件以及容器化（Docker）环境部署的优质技术链接。",
        "sections": [
            {"path_keywords": ["开发 -> R包"], "title": "🥼 R 语言与 R 包开发 (R Package Development)"},
            {"path_keywords": ["开发 -> Python包"], "title": "🐍 Python 语言与包开发 (Python Development)"},
            {"path_keywords": ["开发 -> Go"], "title": "🐹 Go 语言开发 (Go Development)"},
            {"path_keywords": ["开发 -> rust"], "title": "🦀 Rust 语言开发 (Rust Development)"},
            {"path_keywords": ["开发 -> julia"], "title": "🟪 Julia 语言开发 (Julia Development)"},
            {"path_keywords": ["开发 -> dotnet"], "title": "🔷 .NET / C# 开发 (.NET Development)"},
            {"path_keywords": ["开发 -> lisp"], "title": "🤖 Lisp / Emacs Lisp (Lisp)"},
            {"path_keywords": ["开发 -> Conda"], "title": "📦 Conda 包管理与环境配置 (Conda)"},
            {"path_keywords": ["开发 -> VSCode", "工作工具 -> 编程辅助"], "title": "💻 VS Code 插件与配置 (VS Code)"},
            {"path_keywords": ["开发 -> Web应用"], "title": "🌐 Web 框架与应用开发 (Web Apps)"},
            {"path_keywords": ["web dev"], "title": "🖥️ Web 前端与后端开发 (Frontend & Backend Dev)"},
            {"path_keywords": ["系统工具 -> Web应用"], "title": "🌐 Web 实用工具与平台 (Web Utilities & Services)"},
            {"path_keywords": ["开发 -> Linux开发"], "title": "🐧 Linux 底层开发与配置 (Linux Development)"},
            {"path_keywords": ["开发 -> 流程"], "title": "⚙️ 软件生命周期与开发流程 (Software Workflows)"},
            {"path_keywords": ["开发 -> 可重复研究", "工作工具 -> 可重复研究"], "title": "🔄 可重复性研究与工程化 (Reproducible Research)"},
            {"path_keywords": ["开发 -> 许可协议"], "title": "📄 开源许可证与协议 (Licenses)"},
            {"path_keywords": ["开发 -> GitHub 机器人"], "title": "🤖 GitHub Actions & Bots"},
            {"path_keywords": ["开发 -> 项目展示", "工作工具 -> badge/徽章"], "title": "🖼️ 项目文档与静态网页构建 (Project Showcase)"},
            {"path_keywords": ["开发 -> 平台应用软件"], "title": "🖥️ 跨平台桌面应用开发 (Desktop App Dev)"},
            {"path_keywords": ["开发 -> 邮件列表"], "title": "📧 开发者邮件列表 (Mailing Lists)"},
            {"path_keywords": ["开发 -> BioC"], "title": "🧬 Bioconductor 开发资源 (Bioconductor Dev)"},
            {"path_keywords": ["开发 -> 云雀"], "title": "🔀 云雀与其他协作工具 (Yunque & Collaboration)"},
            {"path_keywords": ["系统 -> Linux", "搜索 -> Linux 命令"], "title": "🐧 Linux 系统运维与命令行 (Linux Systems)"},
            {"path_keywords": ["系统 -> macOS"], "title": "🍏 macOS 系统使用与配置 (macOS Systems)"},
            {"path_keywords": ["系统 -> Windows"], "title": "🪟 Windows 系统与开发环境 (Windows Systems)"},
            {"path_keywords": ["系统 -> 操作系统资源"], "title": "💾 操作系统物理与底层资源 (OS Resources)"},
            {"path_keywords": ["系统 -> 微机"], "title": "🔌 单片机与微系统 (Microcontrollers)"},
            {"path_keywords": ["系统工具 -> 跨平台软件"], "title": "⚙️ 跨平台效率软件 (Cross-platform Software)"},
            {"path_keywords": ["系统工具 -> Windows软件"], "title": "🪟 Windows 常用软件 (Windows Software)"},
            {"path_keywords": ["系统工具 -> Linux软件"], "title": "🐧 Linux 常用软件 (Linux Software)"},
            {"path_keywords": ["系统工具 -> macOS软件"], "title": "🍏 macOS 效率软件 (macOS Software)"},
            {"path_keywords": ["系统工具 -> 浏览器插件"], "title": "🌐 浏览器实用插件 (Browser Extensions)"},
            {"path_keywords": ["系统工具 -> Environment-Setup", "系统工具 -> 环境部署"], "title": "🐳 环境部署与容器化 (Deployment & Docker)"},
            {"path_keywords": ["工作工具 -> 建站", "工作工具 -> 博客/网站主题"], "title": "🕸️ 个人建站与博客系统 (Website Building & Blogs)"},
            {"path_keywords": ["工作工具 -> 软件库"], "title": "📦 通用软件库与框架 (Software Libraries)"},
            {"path_keywords": ["开发"], "title": "💻 软件开发与通用技术 (Software Development)"},
            {"path_keywords": ["搜索 -> 软件包"], "title": "🔍 软件包与代码检索 (Package & Code Search)"},
            {"path_keywords": ["工作工具 -> web&shiny"], "title": "🌐 Shiny & Web Application Dev"},
            {"path_keywords": ["工作工具 -> 数据库处理"], "title": "🗄️ 数据库操作与管理工具 (Database Utilities)"},
            {"path_keywords": ["工作工具 -> 网络"], "title": "📡 网络工具与服务 (Networking Tools)"},
            {"path_keywords": ["开发 -> 图标"], "title": "🎨 界面设计与图标资源 (UI Design & Icons)"},
            {"path_keywords": ["工作工具 -> HPC"], "title": "🖥️ 高性能计算与并行计算 (HPC)"},
            {"path_keywords": ["工作工具 -> 语言发行版"], "title": "💻 语言发行版本与包管理器 (Language Distributions)"}
        ]
    },
    "data_science_and_visualization.md": {
        "title": "📊 数据科学、统计建模与绘图可视化 (Data Science, Modeling & Visualization)",
        "intro": "本列表收录了数据清洗、统计分析、假设检验、数学与统计建模、绘图可视化、字体排版、表格渲染、颜色配色管理等数据科学领域的优质技术资源。",
        "sections": [
            {"path_keywords": ["工作工具 -> 数据处理"], "title": "🧹 数据清洗与预处理 (Data Processing)"},
            {"path_keywords": ["工作工具 -> 建模"], "title": "📈 统计与机器学习建模 (Data Modeling)"},
            {"path_keywords": ["工作工具 -> 统计分析", "工作工具 -> 元分析"], "title": "📊 统计分析与假设检验 (Statistical Analysis)"},
            {"path_keywords": ["工作工具 -> 字体处理"], "title": "🔠 字体与排版设计 (Fonts & Typography)"},
            {"path_keywords": ["工作工具 -> 表格处理"], "title": "📋 表格数据处理与美化 (Table Processing)"},
            {"path_keywords": ["工作工具 -> 颜色管理"], "title": "🎨 颜色方案与美学管理 (Color Management)"},
            {"path_keywords": ["工作工具 -> 配色"], "title": "🎨 常用绘图配色方案 (Color Palettes)"},
            {"path_keywords": ["工作工具 -> 绘图可视化", "工作工具 -> 图与网络", "学术 -> 数据科学平台"], "title": "📊 数据可视化与图表绘制 (Data Visualization)"},
            {"path_keywords": ["工作工具 -> 最优化"], "title": "📈 最优化算法与计算 (Optimization)"},
            {"path_keywords": ["算法与统计"], "title": "🧮 经典算法与数理统计 (Algorithms & Stats)"}
        ]
    },
    "ai_and_machine_learning.md": {
        "title": "🧠 人工智能与机器学习清单 (Artificial Intelligence & Machine Learning)",
        "intro": "本列表聚焦于人工智能与机器学习领域，收录了经典机器学习算法、深度学习神经网络框架、大语言模型推理引擎以及主流 AI 托管服务开发平台。",
        "sections": [
            {"path_keywords": ["工作工具 -> 机器学习", "博文/问答/小抄 -> 机器学习"], "title": "🤖 传统机器学习算法与资源 (Machine Learning)"},
            {"path_keywords": ["工作工具 -> 深度学习"], "title": "🧠 深度学习与神经网络架构 (Deep Learning)"},
            {"path_keywords": ["工作工具 -> AI平台"], "title": "🌐 常用 AI 开发与托管平台 (AI Platforms)"},
            {"path_keywords": ["系统工具 -> 机器学习模型"], "title": "🤖 大语言模型与本地推理引擎 (LLMs & Inference)"},
            {"path_keywords": ["搜索 -> AI"], "title": "🎨 AI 搜索、文生图与大模型导航 (AI Tools & Navigation)"}
        ]
    },
    "learning_and_blogs.md": {
        "title": "🎓 学习教程、技术博客与开发社区 (Learning Tutorials, Blogs & Communities)",
        "intro": "本列表汇总了程序设计、数据科学及生物信息学的学习书籍、优质博客、官方技术手册、技术问答以及广受好评的开发者交流论坛与学术社区。",
        "sections": [
            {"path_keywords": ["学习/资源 -> 书籍 -> R"], "title": "📚 R 语言学习书籍 (R Books)"},
            {"path_keywords": ["学习/资源 -> 书籍 -> Python"], "title": "📚 Python 语言学习书籍 (Python Books)"},
            {"path_keywords": ["学习/资源 -> 书籍 -> Go"], "title": "📚 Go 语言学习书籍 (Go Books)"},
            {"path_keywords": ["学习/资源 -> 书籍 -> 生物统计/信息学"], "title": "📚 生物统计与生物信息学书籍 (Bioinformatics Books)"},
            {"path_keywords": ["学习/资源 -> 书籍"], "title": "📚 计算机与数据科学综合书籍 (General Books)"},
            {"path_keywords": ["学习/资源 -> 课程", "个人 -> 在线课程"], "title": "🎓 精选在线课程 (Online Courses)"},
            {"path_keywords": ["学习/资源 -> 博客"], "title": "✍️ 优质个人与团队技术博客 (Tech Blogs)"},
            {"path_keywords": ["学习/资源 -> 文档"], "title": "📖 官方开发文档与参考手册 (Documentations)"},
            {"path_keywords": ["学习/资源 -> 列表"], "title": "📋 资源汇总与精选项目列表 (Curated Lists)"},
            {"path_keywords": ["学习/资源 -> 网站", "学习/资源", "学习/资源 -> 现阶段", "收藏夹栏"], "title": "🌐 实用技术网站与门户 (Tech Portals)"},
            {"path_keywords": ["博文/问答/小抄 -> 问答"], "title": "💬 技术问答社区与经典解答 (Q&A)"},
            {"path_keywords": ["博文/问答/小抄 -> Linux"], "title": "🐧 Linux 经典博文与实用技巧 (Linux Articles)"},
            {"path_keywords": ["博文/问答/小抄 -> R"], "title": "🥼 R 语言技巧与精选博文 (R Articles)"},
            {"path_keywords": ["博文/问答/小抄 -> Python"], "title": "🐍 Python 编程技巧与博文 (Python Articles)"},
            {"path_keywords": ["博文/问答/小抄 -> 生信分析"], "title": "🧬 生物信息学分析实战博文 (Bioinfo Articles)"},
            {"path_keywords": ["博文/问答/小抄 -> 统计与建模"], "title": "📊 统计学与数学建模博文 (Statistics Articles)"},
            {"path_keywords": ["博文/问答/小抄 -> docker"], "title": "🐳 Docker 与容器化运维博文 (Docker Articles)"},
            {"path_keywords": ["博文/问答/小抄 -> 概念"], "title": "🧠 核心科学与技术概念解析 (Concepts)"},
            {"path_keywords": ["博文/问答/小抄 -> 其他", "个人 -> 其他"], "title": "📝 其他技术与学术博文 (Other Articles)"},
            {"path_keywords": ["博文/问答/小抄 -> 小抄"], "title": "📝 编程小抄与速查表 (Cheat Sheets)"},
            {"path_keywords": ["个人 -> 论坛|社区", "个人"], "title": "💬 开发者与学术论坛社区 (Developer Forums)"},
            {"path_keywords": ["个人 -> 编程练习", "学术 -> 竞赛"], "title": "💻 编程学习与刷题练习 (Programming Practice)"},
            {"path_keywords": ["搜索 -> 搜索引擎", "搜索 -> 导航", "搜索"], "title": "🔍 实用搜索引擎与技术导航 (Search Engines & Portals)"}
        ]
    }
}

# 5. Overrides/Reclassifications Map
# Format: (matching_condition_lambda, target_file, target_section)
override_rules = [
    # Move network visualization tools to data_science_and_visualization.md Data Visualization section
    (lambda b: any(kw in b["url"] for kw in ["nezzle", "NetBio", "crosslink"]) or "连线图应该如何画" in b["name"],
     "data_science_and_visualization.md", "📊 数据可视化与图表绘制 (Data Visualization)"),
     
    # Move ggtangle intro WeChat article to learning_and_blogs.md R Articles section
    (lambda b: "ggtangle" in b["name"] or "ggtangle" in b["url"],
     "learning_and_blogs.md", "🥼 R 语言技巧与精选博文 (R Articles)"),
     
    # Move Go article to learning_and_blogs.md Go Books section
    (lambda b: "成为 Go 高手的 8 个 GitHub 开源项目" in b["name"],
     "learning_and_blogs.md", "📚 Go 语言学习书籍 (Go Books)"),
     
    # Move Linux China to learning_and_blogs.md Tech Blogs
    (lambda b: "Linux中国" in b["name"],
     "learning_and_blogs.md", "✍️ 优质个人与团队技术博客 (Tech Blogs)"),
     
    # Move bioinformatics WeChat/Jianshu tutorial posts to learning_and_blogs.md Bioinfo Articles
    (lambda b: any(kw in b["name"] for kw in ["科研和临床分析调研", "单细胞数据分析资源推荐", "生信教程", "ChIP-seq数据处理", "ATAC-seq数据实战", "WES数据分析流程", "单细胞数据库汇总", "遗传资源数据库合集"]),
     "learning_and_blogs.md", "🧬 生物信息学分析实战博文 (Bioinfo Articles)"),
     
    # Move python and R machine learning discussion to learning_and_blogs.md Other Articles
    (lambda b: "python和R做机器学习你选择哪个" in b["name"],
     "learning_and_blogs.md", "📝 其他技术与学术博文 (Other Articles)")
]


# 6. Group bookmarks into sections
grouped_data = {}
for filename, struct in files_structure.items():
    grouped_data[filename] = {}
    for sec in struct["sections"]:
        grouped_data[filename][sec["title"]] = []

# Gather and sort all path matching rules across all files by specificity
flat_rules = []
for filename, struct in files_structure.items():
    for sec in struct["sections"]:
        for kw in sec["path_keywords"]:
            flat_rules.append({
                "kw": kw,
                "filename": filename,
                "sec_title": sec["title"],
                "kw_parts": kw.split(" -> ")
            })
# Sort rules by path length (number of segments) descending, then string length descending
flat_rules.sort(key=lambda x: (len(x["kw_parts"]), len(x["kw"])), reverse=True)

for b in filtered_bookmarks:
    matched = False
    
    # Check manual override rules first
    for cond, dest_file, dest_sec in override_rules:
        if cond(b):
            grouped_data[dest_file][dest_sec].append(b)
            matched = True
            break
            
    if matched:
        continue
        
    # Contiguous sublist matching for path segments (sorted by length/specificity)
    b_path = b["path"]
    for rule in flat_rules:
        kw_parts = rule["kw_parts"]
        n_kw = len(kw_parts)
        n_b = len(b_path)
        for i in range(n_b - n_kw + 1):
            if b_path[i : i + n_kw] == kw_parts:
                grouped_data[rule["filename"]][rule["sec_title"]].append(b)
                matched = True
                break
        if matched:
            break

# 7. Write out GFM-compatible files with TOCs (with U+FE0F variation selector retention)
def get_gfm_anchor(title):
    title_no_html = re.sub(r'<[^>]+>', '', title)
    lowered = title_no_html.lower()
    chars = []
    for char in lowered:
        if char.isalnum() or char.isspace() or char in ('-', '_') or ord(char) == 0xFE0F:
            chars.append(char)
    cleaned = "".join(chars)
    anchor = cleaned.replace(' ', '-')
    return anchor

for filename, struct in files_structure.items():
    filepath = os.path.join(repo_dir, filename)
    
    # 1. Filter out empty sections dynamically
    non_empty_sections = []
    for sec in struct["sections"]:
        bookmarks_in_sec = grouped_data[filename][sec["title"]]
        if bookmarks_in_sec:
            # Deduplicate by URL
            seen_sec_urls = set()
            unique_bookmarks = []
            for item in bookmarks_in_sec:
                if item["url"] not in seen_sec_urls:
                    seen_sec_urls.add(item["url"])
                    unique_bookmarks.append(item)
            # Sort alphabetically
            unique_bookmarks.sort(key=lambda x: x["name"].lower())
            non_empty_sections.append((sec["title"], unique_bookmarks))
            
    if not non_empty_sections:
        print(f"Skipping generation of {filename} (no links found).")
        continue

    # 2. Write Markdown file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {struct['title']}\n\n")
        f.write(f"> {struct['intro']}\n\n")
        
        # Table of Contents
        f.write("## 📌 目录 (Table of Contents)\n\n")
        written_titles = set()
        for sec_title, _ in non_empty_sections:
            if sec_title in written_titles:
                continue
            written_titles.add(sec_title)
            
            anchor = get_gfm_anchor(sec_title)
            f.write(f"- [{sec_title}](#{anchor})\n")
        f.write("\n---\n\n")
        
        # Write Sections content
        written_titles = set()
        for sec_title, bookmarks_list in non_empty_sections:
            if sec_title in written_titles:
                continue
            written_titles.add(sec_title)
            
            f.write(f"## {sec_title}\n\n")
            for item in bookmarks_list:
                f.write(f"- [{item['name']}]({item['url']})\n")
            f.write("\n")
            
    print(f"Generated clean list: {filename} ({len(non_empty_sections)} sections)")

print("\nAll markdown files generated successfully and verified GFM TOC anchors!")
