import laya

state = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice",
    "body": "Hi, we were billed twice. Please refund the duplicate charge."
}



questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else"
        }
    },

    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?"
    }
}


agent = laya.load("./model")

#agent判断
result = agent.predict(state, questions)

print(result)