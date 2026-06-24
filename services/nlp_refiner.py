import os
import dotenv
from openai import OpenAI

NLP_MODEL_NAME = os.getenv("NLP_MODEL_NAME", "openai/gpt-4o-mini")
OPENROUTER_REFERER = os.getenv("OPENROUTER_REFERER", "https://your-app-name.com")
OPENROUTER_TITLE = os.getenv("OPENROUTER_TITLE", "SignToEgyptianApp")

dotenv.load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or dotenv.get_key(dotenv.find_dotenv(), "OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
) if OPENROUTER_API_KEY else None


def refine_to_egyptian(gloss_input: str, model_name: str = NLP_MODEL_NAME) -> str:
    gloss_input = str(gloss_input).strip()
    if not gloss_input:
        return ""

    # Deployment-safe fallback: if key is not configured, return raw predicted words.
    if client is None:
        return gloss_input

    system_prompt = (
        "أنت خبير في تحويل 'كلمات لغة الإشارة' (Glosses) إلى جمل مصرية عامية طبيعية 100%.\n"
        "قواعد العمل:\n"
        "1. سد الفجوات: إذا كانت الجملة ناقصة، استنتج التكملة الأكثر منطقية أو صغها كجملة تامة.\n"
        "2. تصريف الأفعال: لغة الإشارة غالباً تستخدم المصدر، أنت يجب أن تصرف الفعل بناءً على الضمير.\n"
        "3. حروف الجر والروابط: أضف حروف الجر والروابط لتصبح الجملة مفهومة.\n"
        "4. الروح المصرية: استخدم كلمات عامية شائعة مثل عشان، إمبارح، دلوقتي، فين، مش.\n"
        "5. التعامل مع النقص: إذا كان الإدخال كلمات متقاطعة، رتبها لتعطي معنى مفيد.\n"
        "6. لا تغيّر المعنى الأساسي للكلمات المتوقعة من موديل الإشارة إلا عند الضرورة.\n"
        "7. المخرجات: أخرج الجملة النهائية فقط بالعامية المصرية، بدون أي شرح أو مقدمات."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "انا - مدرسه"},
        {"role": "assistant", "content": "أنا رايح المدرسة."},
        {"role": "user", "content": "خالة - باب"},
        {"role": "assistant", "content": "خالتي بتخبط على الباب."},
        {"role": "user", "content": "هو - بيت"},
        {"role": "assistant", "content": "هو رايح البيت."},
        {"role": "user", "content": "انا - شغل - بكرة"},
        {"role": "assistant", "content": "أنا هروح الشغل بكرة."},
        {"role": "user", "content": "هي - اكل"},
        {"role": "assistant", "content": "هي بتاكل."},
        {"role": "user", "content": gloss_input},
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
        extra_headers={
            "HTTP-Referer": OPENROUTER_REFERER,
            "X-Title": OPENROUTER_TITLE,
        },
    )
    return response.choices[0].message.content.strip()
