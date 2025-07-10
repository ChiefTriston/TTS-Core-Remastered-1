def __init__(self, config: Dict[str, Any]) -> None:
    """
    Args:
        config["audio_root"]        (str): path to root “audio” folder with speaker subfolders
        config["transcript_root"]   (str): path to root “transcripts” folder
        config["include_speakers"]  (List[str], optional): exact speaker folder names to load
        config["sample_rate"]       (int, optional): target sampling rate (default: 22050)
        config["resample"]          (bool, optional): whether to resample (default: True)
        config["mel_transform"]     (callable, optional): custom mel transform
        config["frontend"]          (Dict[str,int]): PhonemeFrontend params
        config["text_encoder"]      (str): Sentence-Transformer model ID
    """
    # Roots
    self.audio_root      = Path(config["audio_root"])
    self.transcript_root = Path(config["transcript_root"])
    if not self.audio_root.is_dir():
        raise FileNotFoundError(f"audio_root not found: {self.audio_root}")
    if not self.transcript_root.is_dir():
        raise FileNotFoundError(f"transcript_root not found: {self.transcript_root}")

    # Audio settings
    self.sample_rate      = config.get("sample_rate", 22050)
    self.resample_enabled = config.get("resample", True)
    self.resampler        = None
    self.mel_transform    = config.get("mel_transform", None)

    # Text processing
    self.frontend     = PhonemeFrontend(config.get("frontend", {}))
    self.text_encoder = TextEncoder(config.get("text_encoder",
                                              "sentence-transformers/all-MiniLM-L6-v2"))

    # Auto-discover all speaker folders
    speaker_dirs = [p.name for p in self.audio_root.iterdir() if p.is_dir()]
    # Optionally filter to only include specified speakers
    if config.get("include_speakers"):
        include = set(config["include_speakers"])
        speaker_dirs = [s for s in speaker_dirs if s in include]
        logger.info(f"Including only speakers: {speaker_dirs}")

    self.spk2id = {spk: idx for idx, spk in enumerate(sorted(speaker_dirs))}
    logger.info(f"Registered speakers: {list(self.spk2id.keys())}")

    # (Optional) auto-discover domains/styles if nested under speaker folders
    all_domains = set()
    all_styles  = set()
    for spk in speaker_dirs:
        base = self.audio_root / spk
        for dom in (base.iterdir() if base.exists() else []):
            if dom.is_dir():
                all_domains.add(dom.name)
                for sty in dom.iterdir():
                    if sty.is_dir():
                        all_styles.add(sty.name)
    self.dom2id = {d: i for i, d in enumerate(sorted(all_domains))}
    self.sty2id = {s: i for i, s in enumerate(sorted(all_styles))}

    # Gather and validate (wav, txt, parts)
    self.items = []
    for wav_path in sorted(self.audio_root.rglob("*.wav")):
        rel      = wav_path.relative_to(self.audio_root)
        speaker  = rel.parts[0]
        if speaker not in self.spk2id:
            continue  # skip any speakers not in include_speakers
        txt_path = (self.transcript_root / rel).with_suffix(".txt")
        if txt_path.is_file():
            self.items.append((wav_path, txt_path, rel.parts))
        else:
            logger.warning(f"Missing transcript: {wav_path}")
    if not self.items:
        raise RuntimeError("No valid (wav, txt) pairs found for the specified speakers.")
    logger.info(f"Loaded {len(self.items)} examples.")
