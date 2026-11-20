"""Telegram bot: send a shelf photo, get it back with every product boxed and counted.

Runs anywhere with Python (long polling, no public URL needed). The token comes
from SHELF_BOT_TOKEN in the environment or in the repo's .env file.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from shelf_vision.counter import ShelfCounter, draw
from shelf_vision.env import REPO_ROOT

log = logging.getLogger("shelf_vision.bot")

DEFAULT_CONF = 0.4  # lowest count error on SKU-110K val
MAX_SIDE = 2560  # reply photo size; Telegram rescales anything larger anyway

START_TEXT = (
    "Привет! Пришлите фото полки магазина, и я посчитаю товары на нём.\n\n"
    "• Снимайте полку целиком, прямо и при хорошем свете.\n"
    "• Для лучшей точности отправляйте фото файлом (скрепка → Файл): "
    "так Telegram его не сжимает.\n"
    "• /conf 0.3 меняет порог уверенности: ниже находит больше товаров, "
    "но и больше лишних рамок. По умолчанию 0.4.\n\n"
    "Я считаю товары, но пока не различаю, какой именно это товар. "
    "Ошибка подсчёта на тестовых данных около 9%."
)


def load_token() -> str:
    """SHELF_BOT_TOKEN from the environment, else from the repo's .env file."""
    token = os.environ.get("SHELF_BOT_TOKEN", "")
    env_file = REPO_ROOT / ".env"
    if not token and env_file.exists():
        for line in env_file.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "SHELF_BOT_TOKEN":
                token = value.strip().strip("'\"")
    if not token:
        raise SystemExit(
            "No bot token: put SHELF_BOT_TOKEN=<token from @BotFather> into "
            f"{env_file} or the environment"
        )
    return token


def parse_allowed(value: str) -> set[int]:
    """Telegram user ids from a comma-separated list; empty means everyone."""
    return {int(v) for v in value.replace(" ", "").split(",") if v}


def parse_conf(args: list[str]) -> float | None:
    """Confidence from `/conf <value>`, accepting a comma decimal; None if invalid."""
    if len(args) != 1:
        return None
    try:
        conf = float(args[0].replace(",", "."))
    except ValueError:
        return None
    return conf if 0.05 <= conf <= 0.95 else None


def decode_image(data: bytes) -> np.ndarray:
    """BGR array from image bytes, with EXIF rotation applied (photos sent as files keep it)."""
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(data)) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    return np.ascontiguousarray(rgb[:, :, ::-1])


def encode_jpeg(bgr: np.ndarray) -> bytes:
    import cv2

    h, w = bgr.shape[:2]
    if max(h, w) > MAX_SIDE:
        r = MAX_SIDE / max(h, w)
        bgr = cv2.resize(bgr, (round(w * r), round(h * r)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError("could not encode the reply photo")
    return buf.tobytes()


def caption(n: int, conf: float, seconds: float) -> str:
    return f"Товаров на фото: {n}\nпорог уверенности {conf:.2f} · {seconds:.1f} с"


def run(model: Path, allowed: set[int], save_dir: Path | None, threads: int = 0) -> None:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

    counter = ShelfCounter(model, threads=threads)
    lock = asyncio.Lock()  # one photo at a time: inference already uses every core

    def permitted(update: Update) -> bool:
        return not allowed or (update.effective_user and update.effective_user.id in allowed)

    async def deny(update: Update) -> None:
        uid = update.effective_user.id if update.effective_user else "?"
        await update.effective_message.reply_text(
            f"У вас нет доступа к этому боту. Ваш Telegram ID: {uid}"
        )

    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not permitted(update):
            return await deny(update)
        await update.effective_message.reply_text(START_TEXT)

    async def set_conf(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not permitted(update):
            return await deny(update)
        conf = parse_conf(context.args or [])
        if conf is None:
            current = context.chat_data.get("conf", DEFAULT_CONF)
            await update.effective_message.reply_text(
                f"Сейчас порог {current:.2f}. Пример: /conf 0.3 (от 0.05 до 0.95)"
            )
            return
        context.chat_data["conf"] = conf
        await update.effective_message.reply_text(f"Порог уверенности: {conf:.2f}")

    async def on_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not permitted(update):
            return await deny(update)
        msg = update.effective_message
        tg_file = await (msg.photo[-1] if msg.photo else msg.document).get_file()
        data = bytes(await tg_file.download_as_bytearray())
        conf = context.chat_data.get("conf", DEFAULT_CONF)
        await msg.chat.send_action("upload_photo")
        try:
            bgr = await asyncio.to_thread(decode_image, data)
        except (OSError, ValueError):  # PIL: unknown or broken image (e.g. HEIC)
            await msg.reply_text("Не получилось открыть картинку. Пришлите фото в JPEG или PNG.")
            return
        async with lock:
            t = time.perf_counter()
            boxes, _ = await asyncio.to_thread(counter.detect, bgr, conf)
            seconds = time.perf_counter() - t
        reply = await asyncio.to_thread(lambda: encode_jpeg(draw(bgr, boxes)))
        await msg.reply_photo(reply, caption=caption(len(boxes), conf, seconds))
        log.info(
            "user %s: %d products, conf %.2f, %.1fs", msg.from_user.id, len(boxes), conf, seconds
        )
        if save_dir is not None:
            save(save_dir, data, msg.from_user.id, len(boxes), conf)

    app = Application.builder().token(load_token()).concurrent_updates(True).build()
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(CommandHandler("conf", set_conf))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, on_image))
    log.info(
        "bot running (model %s, %s)",
        model,
        f"{len(allowed)} allowed users" if allowed else "open to everyone",
    )
    app.run_polling(allowed_updates=Update.ALL_TYPES)


def save(save_dir: Path, data: bytes, user_id: int, count: int, conf: float) -> None:
    """Keep the original photo and its count, one folder per day (for later evaluation)."""
    now = datetime.now()
    day = save_dir / now.strftime("%Y-%m-%d")
    day.mkdir(parents=True, exist_ok=True)
    stem = f"{now.strftime('%H%M%S_%f')}_{user_id}"
    ext = ".png" if data.startswith(b"\x89PNG") else ".jpg"
    (day / f"{stem}{ext}").write_bytes(data)
    meta = {
        "user": user_id,
        "count": count,
        "conf": conf,
        "time": now.isoformat(timespec="seconds"),
    }
    (day / f"{stem}.json").write_text(json.dumps(meta))
