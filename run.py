import argparse
import os
import json
import time
import torch
from model_manager import ModelManager
from memory import MemoryIndex
from utils import export_mind

DATA_DIR = "data"
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")
CHECKPOINT_INITIAL = os.path.join(DATA_DIR, "checkpoint_initial")

def ensure_data_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

def write_empty_file(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        open(path, "w", encoding="utf-8").close()

def load_or_create_metadata():
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
                if "baby_name" in meta:
                    return meta
        except Exception:
            pass
    print("Parece que es la primera vez que ejecutas este proyecto en este directorio.")
    baby_name = input("Dale un nombre al Hijo Digital (por ejemplo: 'Tomás'): ").strip()
    if not baby_name:
        baby_name = "Hijo"
    meta = {
        "baby_name": baby_name,
        "created_at": time.time(),
        "seed": int(time.time())
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Nombre guardado: {baby_name}. Para crear otro Hijo Digital, copia este directorio del proyecto a otra carpeta.")
    return meta

def append_history(history_path, obj):
    with open(history_path, "a", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
        f.write("\n")

def initialize_everything_if_needed(cfg, model_name, device):
    history_path = cfg.get("history_path", "data/history.jsonl")
    memories_store = cfg.get("memories_store", "data/memories.jsonl")
    faiss_path = cfg.get("faiss_index_path", "data/faiss")

    write_empty_file(history_path)
    write_empty_file(memories_store)

    mem = MemoryIndex(embeddings_model_name=cfg.get("embeddings_model", "all-MiniLM-L6-v2"),
                      index_path=faiss_path)
    if not (os.path.exists(faiss_path + ".index") and os.path.exists(faiss_path + ".texts.jsonl")):
        print("[INFO] Creando índice de memoria FAISS vacío y archivos iniciales...")
        mem.save(faiss_path)
        open(faiss_path + ".texts.jsonl", "w", encoding="utf-8").close()
    else:
        try:
            mem.load(faiss_path)
        except Exception:
            mem = MemoryIndex(cfg.get("embeddings_model", "all-MiniLM-L6-v2"), index_path=faiss_path)
            mem.save(faiss_path)

    # Create initial checkpoint only once (downloads model+tokenizer to HF cache the first time)
    if not os.path.exists(CHECKPOINT_INITIAL):
        print("[INFO] Creando checkpoint inicial del modelo (se descargará modelo/tokenizer si es necesario)...")
        mm = ModelManager(model_name=model_name, device=device, gen_cfg=cfg.get("generation", None))
        mm.save_checkpoint(CHECKPOINT_INITIAL, metadata={"created_at": time.time(), "note": "initial checkpoint"})
        print(f"[INFO] Checkpoint inicial guardado en {CHECKPOINT_INITIAL}")
    return mem

def main():
    parser = argparse.ArgumentParser(description="Hijo Digital - Interacción por línea (espera siempre al usuario/padre)")
    parser.add_argument("--gpu", action="store_true", help="Forzar intento de usar GPU si está disponible")
    parser.add_argument("--model", type=str, default=None, help="Modelo base a usar (override config)")
    parser.add_argument("--config", type=str, default="config.json", help="Ruta a config JSON")
    parser.add_argument("--export-mind", action="store_true", help="Exportar la mente (data/) a zip")
    args = parser.parse_args()

    ensure_data_dirs()
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {}

    metadata = load_or_create_metadata()
    baby_name = metadata.get("baby_name", "Hijo")

    use_gpu_cfg = cfg.get("use_gpu", False)
    use_gpu = args.gpu or use_gpu_cfg
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Usando GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if use_gpu and not torch.cuda.is_available():
            print("[WARN] Se pidió usar GPU pero no hay CUDA disponible. Ejecutando en CPU.")
        else:
            print("[INFO] Ejecutando en CPU (cambia use_gpu en config.json a true para intentar usar GPU por defecto).")

    model_name = args.model or cfg.get("model_name", "distilgpt2")

    mem = initialize_everything_if_needed(cfg, model_name, device)

    # Prefer loading from initial checkpoint to avoid re-downloads
    mm = ModelManager(model_name=model_name, device=device, gen_cfg=cfg.get("generation", None))
    if os.path.exists(CHECKPOINT_INITIAL):
        try:
            mm.load_checkpoint(CHECKPOINT_INITIAL, device=device)
        except Exception:
            pass

    history_path = cfg.get("history_path", "data/history.jsonl")
    memories_store = cfg.get("memories_store", "data/memories.jsonl")
    retrieve_k = cfg.get("retrieve_k", 4)

    # Curiosity settings: threshold empiric; puedes ajustarlo en configuración si quieres
    CURIOSITY_ENTROPY_THRESHOLD = 4.0  # valores típicos varían por vocab; ajústalo si es necesario
    pending_teach = False  # cuando es True, la próxima entrada del usuario se guarda como memoria automática

    print(f"\n*** HIJO DIGITAL '{baby_name}': línea de comandos interactiva (espera siempre por la entrada) ***")
    print("Comandos en una línea: /salir  /guardar  /enseñar <texto>  /recuerdos  /checkpoint  /exportar")
    print("Modo normal: escribe tu línea y presiona Enter; el Hijo Digital responderá y, si tiene dudas, preguntará algo para aprender.\n")

    while True:
        try:
            text = input("Tú: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[INFO] Interrupción. Guardando memoria y saliendo.")
            mem.save(cfg.get("faiss_index_path", "data/faiss"))
            break

        if not text:
            continue

        # If pending_teach flag is set, treat this input as teaching and store automatically
        if pending_teach and not (text.startswith("/") or text.lower().startswith("enseñar ")):
            teach_text = text
            mem.add([teach_text])
            append_history(history_path, {"type": "auto_teach", "text": teach_text, "ts": time.time()})
            with open(memories_store, "a", encoding="utf-8") as f:
                json.dump({"text": teach_text, "ts": time.time(), "auto": True}, f, ensure_ascii=False)
                f.write("\n")
            mem.save(cfg.get("faiss_index_path", "data/faiss"))
            print(f"[{baby_name}] Gracias — lo guardé como recuerdo (respuesta a mi pregunta).")
            pending_teach = False
            continue

        # Commands
        if text.lower() in ("/salir", "salir", "exit", "/exit"):
            print("[INFO] Guardando memoria y saliendo.")
            mem.save(cfg.get("faiss_index_path", "data/faiss"))
            break
        if text.lower().startswith("/enseñar ") or text.lower().startswith("enseñar "):
            teach_text = text.partition(" ")[2].strip()
            if teach_text:
                mem.add([teach_text])
                append_history(history_path, {"type": "teach", "text": teach_text, "ts": time.time()})
                with open(memories_store, "a", encoding="utf-8") as f:
                    json.dump({"text": teach_text, "ts": time.time()}, f, ensure_ascii=False)
                    f.write("\n")
                mem.save(cfg.get("faiss_index_path", "data/faiss"))
                print(f"[{baby_name}] Gracias, lo he guardado como recuerdo.")
            else:
                print("[ERROR] /enseñar requiere texto después del comando.")
            continue
        if text.lower().strip() in ("/recuerdos", "recuerdos"):
            print(f"[{baby_name}] Recuperando recuerdos relevantes...")
            top = mem.query(text, k=retrieve_k)
            if not top:
                print("  (no hay recuerdos aún)")
            for i, r in enumerate(top, 1):
                print(f"  {i}. {r}")
            continue
        if text.lower().strip() in ("/guardar", "guardar", "/checkpoint"):
            cp_dir = f"checkpoints/checkpoint_{int(time.time())}"
            mm.save_checkpoint(cp_dir, metadata={"baby_name": baby_name, "saved_at": time.time()})
            mem.save(cfg.get("faiss_index_path", "data/faiss"))
            print(f"[INFO] Checkpoint guardado en {cp_dir}")
            continue
        if text.lower().strip() in ("/exportar", "export", "exportar_mind"):
            out, digest, meta = export_mind(DATA_DIR)
            print(f"[EXPORT] {out} sha256: {digest} meta: {meta}")
            continue

        # Default chat flow: retrieve memories -> generate
        retrieved = mem.query(text, k=retrieve_k)
        respuesta = mm.generate_with_context(text, retrieved)
        print(f"{baby_name}: {respuesta}")
        append_history(history_path, {"type": "chat", "user": text, "response": respuesta, "ts": time.time()})

        # Curiosity: estimate entropía y, si es alta, hacer una pregunta para aprender
        try:
            entropy = mm.estimate_next_token_entropy(text, context_texts=retrieved)
            # Debug: puedes comentar la siguiente línea si no quieres ver la entropía en pantalla
            # print(f"[DEBUG] Entropía estimada: {entropy:.4f}")
            if entropy >= CURIOSITY_ENTROPY_THRESHOLD:
                # Genera una pregunta de curiosidad y marca pending_teach=True
                c_question = mm.make_curiosity_question(text)
                print(f"{baby_name} (curioso): {c_question}")
                print(f"(Si respondes la pregunta inmediatamente, lo guardaré como recuerdo automáticamente.)")
                pending_teach = True
        except Exception:
            # Si algo falla en la estimación (por ejemplo long inputs), no bloqueamos la conversación.
            pending_teach = False
            continue

if __name__ == "__main__":
    main()
