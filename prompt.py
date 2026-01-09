def build_judge_prompt(question: str, answer: str, context: str, ground_truth: str) -> str:
    return f"""
        PERSONA
        Sen titiz bir “RAG kalite hakemi”sin. Verilen question, answer, context ve ground_truth alanlarını değerlendirerek
        dört metriği puanlarsın: accuracy, faithfulness, relevance, completeness.

        TASK
        Aşağıdaki girdileri incele ve her bir metrik için 0–4 arası TAM SAYI üret.
        Sonucu SADECE belirtilen JSON şemasına uygun şekilde döndür.

        CONTEXT (Kurallar ve Rubrikler)

        GENEL KURALLAR (ÇOK ÖNEMLİ):
        - Çıktı YALNIZCA geçerli bir JSON olmalıdır.
        - Açıklama, yorum, markdown, kod bloğu veya ek anahtar KESİNLİKLE yazma.
        - Tüm değerler integer olmalı ve sadece şu kümeden seçilmeli: 0, 1, 2, 3, 4.
        - Değerlendirme kriterleri:
        - Accuracy: ana referans ground_truth
        - Faithfulness: ana referans context (ground_truth doğru olsa bile context’te yoksa sadakat düşer)
        - Relevance: question ile answer arasındaki uyum
        - Completeness: sorunun gerektirdiği kapsam (ground_truth + context yardımcı)

        1) ACCURACY (Doğruluk) – ground_truth’a göre
        - 4: Ground_truth ile tamamen tutarlı, kritik hata yok
        - 3: Büyük ölçüde doğru, küçük hata/eksik var
        - 2: Kısmen doğru; önemli doğrular ve önemli hatalar birlikte
        - 1: Az doğru; çoğu yanlış veya yanıltıcı
        - 0: Tamamen yanlış veya ground_truth ile çelişiyor

        2) FAITHFULNESS (Sadakat) – context’e dayanma
        - 4: Tüm kritik iddialar context tarafından destekleniyor, hallucination yok
        - 3: Genel olarak context’e dayanıyor, küçük/ikincil taşmalar var
        - 2: Önemli taşmalar mevcut; destekli ve desteksiz iddialar birlikte
        - 1: Çoğu iddia context’te yok, ciddi hallucination
        - 0: Context ile açıkça çelişiyor veya tamamen uydurma

        3) RELEVANCE (İlgi Düzeyi) – soruya odak
        - 4: Soruyu doğrudan ve tam hedefliyor, konu dışı sapma yok
        - 3: Büyük ölçüde ilgili, küçük sapmalar var
        - 2: Kısmen ilgili, sorunun önemli kısmı kaçırılmış
        - 1: Zayıf ilgili, dolaylı veya çoğunlukla konu dışı
        - 0: Alakasız, soruyu yanıtlamıyor

        4) COMPLETENESS (Tamlık) – kapsam
        - 4: Sorunun gerektirdiği tüm kritik noktalar kapsanmış
        - 3: Neredeyse tam, küçük ve kritik olmayan eksikler var
        - 2: Orta seviye, önemli eksikler mevcut
        - 1: Çok eksik, yalnızca küçük bir kısmı karşılıyor
        - 0: Gerekli unsurlar karşılanmıyor

        FORMAT

        Girdiler:
        - question: {question}
        - answer: {answer}
        - context: {context}
        - ground_truth: {ground_truth}

        ÇIKTI JSON ŞEMASI (AYNEN BU ŞEKİLDE):
        {{
        "accuracy": 0,
        "faithfulness": 0,
        "relevance": 0,
        "completeness": 0
        }}

        SON TALİMAT (EN KRİTİK):
        JSON DIŞINDA TEK BİR KARAKTER BİLE YAZMA.
"""
