def reply(message: str) -> str:
    msg = message.lower().strip()
    if "refund" in msg:
        return "রিফান্ড প্রসেস করতে আপনার অর্ডার আইডি দিন।"
    if "delivery" in msg or "shipping" in msg:
        return "ডেলিভারি সাধারণত ২-৫ কর্মদিবস লাগে। আপনার অর্ডার আইডি দিলে স্ট্যাটাস বলবো।"
    if "size" in msg:
        return "সাইজ গাইড দেখার জন্য প্রোডাক্ট পেজে 'Size Guide' সেকশন দেখুন।"
    return "ধন্যবাদ! আপনি কী বিষয়ে সাহায্য চান—অর্ডার, ডেলিভারি, রিফান্ড, সাইজ?"
