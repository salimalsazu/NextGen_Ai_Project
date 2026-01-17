def reply(message: str) -> str:
    msg = message.lower().strip()

    if "refund" in msg:
        return (
            "We’re happy to help with your refund request.\n\n"
            "Please share your Order ID so we can check the order status. "
            "Refunds are usually processed within 5–7 business days after approval, "
            "and the amount will be credited back to your original payment method."
        )

    if "delivery" in msg or "shipping" in msg:
        return (
            "Thank you for asking about delivery.\n\n"
            "Standard delivery usually takes 2–5 business days, depending on your location. "
            "Once your order is shipped, you will receive a tracking number. "
            "Please provide your Order ID if you’d like us to check the current delivery status."
        )

    if "size" in msg:
        return (
            "Need help choosing the right size?\n\n"
            "You can find detailed measurements in the ‘Size Guide’ section on the product page. "
            "If you’re still unsure, tell us your height, weight, or usual size, "
            "and we’ll be happy to recommend the best option for you."
        )

    if "cancel" in msg or "cancellation" in msg:
        return (
            "Order cancellation is possible before the item is shipped.\n\n"
            "Please share your Order ID as soon as possible, and we’ll check whether "
            "the order can still be canceled."
        )

    if "payment" in msg:
        return (
            "If you’re facing a payment issue, don’t worry—we can help.\n\n"
            "Please confirm whether the payment was deducted and share your Order ID "
            "or transaction reference number so we can investigate."
        )

    return (
        "Thank you for contacting us! 😊\n\n"
        "How can we assist you today?\n"
        "You can ask about:\n"
        "• Order status\n"
        "• Delivery or shipping\n"
        "• Refunds or cancellations\n"
        "• Size guide or product details\n\n"
        "Just type your question and we’ll take it from there!"
    )
