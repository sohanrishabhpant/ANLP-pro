# **ANLP Project: English to Hinglish Translation**

## **Overview**
This project focuses on implementing English-to-Hinglish translation using two distinct methods:

1. **Direct Translation:**  
   Translating directly from English to Hinglish.

2. **Two-Step Translation:**  
   Translating English to Hindi (in Devanagari script) and then converting the Hindi text into Hinglish. This method leverages the intermediary Hindi representation to enhance translation quality.

## **Motivation**
With the increasing use of Hinglish (a mix of Hindi and English written in the Roman script) in digital communication, there is a growing need for efficient and accurate translation tools. This project explores multiple approaches to achieve this goal.

## **Methods**
### 1. **Direct Translation (English to Hinglish)**  
   - A direct mapping approach that translates English text into Hinglish without any intermediate representation.
   - **Example:**  
     - Input: `How are you?`  
     - Output: `Tum kaise ho?`

### 2. **Two-Step Translation (English → Hindi → Hinglish)**  
   - **Step 1:** English text is first translated into Hindi using Devanagari script.  
   - **Step 2:** The Hindi text is transliterated into Hinglish.  
   - **Example:**  
     - Input: `How are you?`  
     - Step 1 Output (Hindi): `तुम कैसे हो?`  
     - Step 2 Output (Hinglish): `Tum kaise ho?`

## **Technologies Used**
- **Language Models:** Utilized pre-trained models for English-to-Hindi and transliteration tasks.
- **Libraries/Frameworks:** 
  - Python
  - Natural Language Toolkit (NLTK)
  - Hugging Face Transformers
  - IndicTrans for Hindi Transliteration

## **Results**
- **Direct Translation:** Achieved effective results for conversational phrases with minimal errors.
- **Two-Step Translation:** Demonstrated improved contextual accuracy by leveraging Hindi as an intermediary representation.

## **Future Work**
- Integrating a broader vocabulary for Hinglish colloquialisms.
- Enhancing transliteration accuracy for complex phrases.
- Exploring neural machine translation models tailored for Hinglish.

## **Contributors**
- [sohan](https://github.com/sohanrishabhpant)


---

Feel free to reach out for questions or collaboration opportunities!
