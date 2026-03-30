import logging
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm  # 建议安装 pip install tqdm 显示进度

from praatio import textgrid
from lhotse import CutSet, load_manifest_lazy
from lhotse.supervision import AlignmentItem

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def index_textgrids(root_dir: Path) -> Dict[str, Path]:
    """
    递归扫描 root_dir 下所有的 .TextGrid 文件，建立 {recording_id: full_path} 的映射。
    这样可以忽略目录结构的差异（如 train-clean-100 vs train_clean_100）。
    """
    logging.info(f"Indexing TextGrids in {root_dir} ... This might take a while.")
    path_map = {}
    
    # rglob 会递归查找所有子目录
    files = list(root_dir.rglob("*.TextGrid")) + list(root_dir.rglob("*.textgrid"))
    
    for p in tqdm(files, desc="Indexing"):
        # p.stem 是文件名不带后缀，例如 '483_125116_000160_000002'
        path_map[p.stem] = p
        
    logging.info(f"Found {len(path_map)} TextGrid files.")
    return path_map

def add_alignments(
    cuts: CutSet, 
    tg_path_map: Dict[str, Path], 
    tier_name: str = "words"
) -> CutSet:
    
    new_cuts = []
    
    # 使用 tqdm 显示处理进度
    for cut in tqdm(cuts.to_eager(), desc="Processing Cuts"):
        
        # === 关键适配点 1: 使用 recording_id 匹配 ===
        # Cut ID: '483_...-93600'
        # Recording ID: '483_125116_000160_000002' (对应文件名)

        # print(f"cut:{cut}")

        rec_id = cut.recording_id
        
        # 如果 cut 没有 recording_id (罕见)，尝试从 supervision 获取
        if not rec_id and len(cut.supervisions) > 0:
            rec_id = cut.supervisions[0].recording_id

        if rec_id not in tg_path_map:
            # 找不到对应的 TextGrid 文件
            # logging.warning(f"TextGrid not found for recording {rec_id} (Cut: {cut.id})")
            new_cuts.append(cut)
            continue
            
        tg_path = tg_path_map[rec_id]
        
        try:
            # === 关键适配点 2: Praatio 加载 ===
            tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)
            
            # 获取层级 (Tier)
            target_tier = tg.getTier(tier_name) # MFA 默认通常叫 'phones'
            
            alignment_items = []
            
            # 遍历 entries (start, end, label)
            for entry in target_tier.entries:
                start, end, label = entry.start, entry.end, entry.label
                
                # 过滤无用符号
                if not label or label in ["", "sil", "sp", "<sil>", "SIL"]:
                    continue
                
                dur = end - start
                if dur <= 1e-5: # 忽略极短的片段
                    continue

                # === 关键适配点 3: 时间偏移 ===
                # 如果这是一个切片(Cut)，其 start > 0。
                # 对齐信息是相对于原始录音(Recording)的。
                # Lhotse 的 AlignmentItem 应该是相对于 Cut 开始的时间吗？
                # Lhotse 规范：supervision.alignment 的时间是相对于 Supervision start 的。
                # 在你的例子中：cut.start=0, supervision.start=0，所以直接用 entry.start 即可。
                # 如果 cut 是切出来的片段，需要减去 supervision.start。
                
                # 你的 Cut 看起来是整句 (start=0)，但为了通用性，我们可以计算相对时间：
                # (注意：如果 supervision.start > 0，这里需要减去它。但通常 MFA 对齐是基于整句音频的)
                # 简单起见，对于 LibriTTS 这种整句切分的数据，直接存原始时间即可，
                # Lhotse 在后续处理时会自动根据 cut.start 做截取。
                
                item = AlignmentItem(
                    symbol=label,
                    start=start, 
                    duration=dur
                )
                alignment_items.append(item)

            # 注入数据
            if len(cut.supervisions) > 0:
                sup = cut.supervisions[0]
                if sup.alignment is None:
                    sup.alignment = {}
                
                sup.alignment['words'] = alignment_items
                
        except Exception as e:
            logging.warning(f"Error processing {tg_path}: {e}")
        
        new_cuts.append(cut)
        
        # print(f"cut:{cut}")
        # while True:
        #     pass

    return CutSet.from_cuts(new_cuts)

if __name__ == "__main__":
    # === 路径配置 ===
    # TextGrid 的根目录，脚本会自动遍历子文件夹
    ALIGNMENT_ROOT = Path("/star-oss/hanzhifeng/streaming/ZipVoice/LibriTTS-alignment/data")
    
    # 你的 Cuts 文件路径
    INPUT_CUTS = Path("/star-oss/hanzhifeng/streaming/ZipVoice/egs/zipvoice/data/fbank/libritts_cuts_dev-clean.jsonl.gz") # 举例
    OUTPUT_CUTS = Path("aligned_data/fbank/libritts_cuts_dev-clean_aligned.jsonl.gz")

    if not INPUT_CUTS.exists(): 
        print(f"File not found: {INPUT_CUTS}")
        exit(1)

    # 1. 建立索引 (解决嵌套目录问题)
    tg_map = index_textgrids(ALIGNMENT_ROOT)
    
    # 2. 加载 Cuts
    logging.info(f"Loading cuts from {INPUT_CUTS}...")
    cuts = load_manifest_lazy(INPUT_CUTS)
    
    # 3. 处理
    logging.info("Aligning...")
    # 注意：MFA 输出的 TextGrid 里的音素层通常叫 "phones"，如果你的不同请修改 tier_name
    cuts_with_align = add_alignments(cuts, tg_map, tier_name="words")
    
    # 4. 保存
    logging.info(f"Saving to {OUTPUT_CUTS}...")
    cuts_with_align.to_file(OUTPUT_CUTS)
    logging.info("Done.")