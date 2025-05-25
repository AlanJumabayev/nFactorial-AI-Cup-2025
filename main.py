#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SafeDocs - AI защита юридических документов
Веб-приложение для анализа договоров с фокусом на казахстанское законодательство
"""

import os
import io
import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Optional, Union
from pathlib import Path

# Web Framework
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.encoders import jsonable_encoder

# Data Models
from pydantic import BaseModel, Field, validator

# File Processing
import PyPDF2
from PIL import Image, ImageEnhance
import pytesseract

# Environment
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

# ============================================================================
# МОДЕЛИ ДАННЫХ
# ============================================================================

class HighlightData(BaseModel):
    """Данные для подсветки текста"""
    start: int = Field(..., description="Начальная позиция")
    end: int = Field(..., description="Конечная позиция")
    type: str = Field(..., description="Тип: risk, benefit, unclear")
    text: str = Field(..., description="Подсвеченный текст")
    category: str = Field(..., description="Категория находки")
    severity: Optional[str] = Field(None, description="Уровень серьезности")

class RiskData(BaseModel):
    """Данные о риске"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str = Field(..., description="Тип риска")
    keyword: str = Field(..., description="Ключевое слово")
    context: str = Field(..., description="Контекст нахождения")
    position: int = Field(..., description="Позиция в тексте")
    severity: str = Field(..., description="Уровень: низкий, средний, высокий")
    recommendation: str = Field(..., description="Рекомендация")
    legal_reference: Optional[str] = Field(None, description="Ссылка на закон")

class BenefitData(BaseModel):
    """Данные о выгоде"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str = Field(..., description="Тип выгоды")
    keyword: str = Field(..., description="Ключевое слово")
    context: str = Field(..., description="Контекст нахождения")
    position: int = Field(..., description="Позиция в тексте")
    value: str = Field(..., description="Ценность: низкая, средняя, высокая")
    description: str = Field(..., description="Описание выгоды")

class UnclearTermData(BaseModel):
    """Данные о неясном термине"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    phrase: str = Field(..., description="Неясная фраза")
    context: str = Field(..., description="Контекст")
    position: int = Field(..., description="Позиция в тексте")
    explanation: str = Field(..., description="Объяснение проблемы")
    suggestion: str = Field(..., description="Предложение по улучшению")
    legal_clarification: Optional[str] = Field(None, description="Юридическое разъяснение")

class DocumentAnalysis(BaseModel):
    """Результат анализа документа"""
    document_id: str = Field(..., description="Уникальный ID документа")
    filename: str = Field(..., description="Имя файла")
    file_type: str = Field(..., description="Тип файла")
    file_size: int = Field(..., description="Размер файла в байтах")
    
    # Текстовые данные
    text_content: str = Field(..., description="Краткое содержание")
    full_text: str = Field(..., description="Полный текст документа")
    word_count: int = Field(..., description="Количество слов")
    
    # Результаты анализа
    risks: List[RiskData] = Field(default_factory=list, description="Найденные риски")
    benefits: List[BenefitData] = Field(default_factory=list, description="Выгодные условия")
    unclear_terms: List[UnclearTermData] = Field(default_factory=list, description="Неясные термины")
    
    # Общая оценка
    overall_rating: str = Field(..., description="Общая оценка безопасности")
    risk_score: float = Field(..., description="Числовая оценка риска (0-100)")
    
    # Дополнительные данные
    summary: str = Field(..., description="Краткое резюме")
    recommendations: List[str] = Field(default_factory=list, description="Рекомендации")
    highlights: List[HighlightData] = Field(default_factory=list, description="Подсветка")
    
    # Метаданные
    processed_at: str = Field(..., description="Время обработки")
    processing_time: float = Field(..., description="Время обработки в секундах")
    version: str = Field(default="1.0.0", description="Версия анализатора")

class ChatMessage(BaseModel):
    """Сообщение в чате"""
    document_id: str = Field(..., description="ID документа")
    question: str = Field(..., max_length=1000, description="Вопрос пользователя")

class ChatResponse(BaseModel):
    """Ответ чата"""
    answer: str = Field(..., description="Ответ AI")
    confidence: float = Field(..., description="Уверенность в ответе")
    relevant_sections: List[str] = Field(default_factory=list, description="Релевантные секции")
    document_id: str = Field(..., description="ID документа")

# ============================================================================
# ОБРАБОТКА ФАЙЛОВ
# ============================================================================

class FileProcessor:
    """Высококачественная обработка различных типов файлов"""
    
    # Поддерживаемые типы файлов
    SUPPORTED_TYPES = {
        'application/pdf': 'pdf',
        'image/jpeg': 'image',
        'image/jpg': 'image',
        'image/png': 'image',
        'text/plain': 'text',
        'application/msword': 'doc',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx'
    }
    
    # Максимальный размер файла (20MB)
    MAX_FILE_SIZE = 20 * 1024 * 1024
    
    @staticmethod
    def validate_file(file: UploadFile) -> Dict[str, any]:
        """Валидация загруженного файла"""
        
        # Проверка типа файла
        if file.content_type not in FileProcessor.SUPPORTED_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Неподдерживаемый тип файла: {file.content_type}. "
                       f"Поддерживаются: {', '.join(FileProcessor.SUPPORTED_TYPES.keys())}"
            )
        
        # Проверка имени файла
        if not file.filename:
            raise HTTPException(status_code=400, detail="Имя файла не указано")
        
        return {
            'file_type': FileProcessor.SUPPORTED_TYPES[file.content_type],
            'content_type': file.content_type,
            'filename': file.filename
        }
    
    @staticmethod
    async def process_file(file: UploadFile) -> Dict[str, any]:
        """Обработка файла и извлечение текста"""
        
        # Валидация
        file_info = FileProcessor.validate_file(file)
        
        # Чтение содержимого
        try:
            file_content = await file.read()
        except Exception as e:
            logger.error(f"Ошибка чтения файла: {e}")
            raise HTTPException(status_code=400, detail="Ошибка чтения файла")
        
        # Проверка размера
        if len(file_content) > FileProcessor.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Файл слишком большой: {len(file_content)} байт. "
                       f"Максимум: {FileProcessor.MAX_FILE_SIZE} байт"
            )
        
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Файл пустой")
        
        # Извлечение текста
        file_type = file_info['file_type']
        
        try:
            if file_type == 'pdf':
                text = FileProcessor._extract_from_pdf(file_content)
            elif file_type == 'image':
                text = FileProcessor._extract_from_image(file_content)
            elif file_type == 'text':
                text = FileProcessor._extract_from_text(file_content)
            else:
                raise HTTPException(status_code=400, detail=f"Обработка {file_type} пока не поддерживается")
            
            # Валидация извлеченного текста
            if not text or len(text.strip()) < 50:
                raise HTTPException(
                    status_code=400,
                    detail="Извлечено слишком мало текста для анализа (минимум 50 символов)"
                )
            
            return {
                'text': text.strip(),
                'file_type': file_type,
                'filename': file_info['filename'],
                'file_size': len(file_content),
                'word_count': len(text.split())
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Ошибка извлечения текста из {file_type}: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка обработки файла: {str(e)}")
    
    @staticmethod
    def _extract_from_pdf(content: bytes) -> str:
        """Извлечение текста из PDF"""
        try:
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            if len(pdf_reader.pages) == 0:
                raise Exception("PDF не содержит страниц")
            
            text_parts = []
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_parts.append(f"\n=== Страница {page_num} ===\n")
                        text_parts.append(page_text)
                        text_parts.append("\n")
                except Exception as e:
                    logger.warning(f"Ошибка извлечения текста со страницы {page_num}: {e}")
                    continue
            
            full_text = "".join(text_parts)
            
            if not full_text.strip():
                raise Exception("PDF не содержит извлекаемого текста")
            
            return full_text.strip()
            
        except Exception as e:
            raise Exception(f"Ошибка обработки PDF: {str(e)}")
    
    @staticmethod
    def _extract_from_image(content: bytes) -> str:
        """Высококачественное OCR для изображений"""
        try:
            # Открытие изображения
            image = Image.open(io.BytesIO(content))
            
            # Конвертация в RGB если нужно
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            
            # Улучшение качества изображения
            image = FileProcessor._enhance_image_for_ocr(image)
            
            # OCR с оптимизированными настройками
            custom_config = r'--oem 3 --psm 6 -l rus+eng'
            
            try:
                text = pytesseract.image_to_string(image, config=custom_config)
            except Exception as ocr_error:
                logger.warning(f"OCR с конфигурацией не удался: {ocr_error}")
                # Fallback - простой OCR
                text = pytesseract.image_to_string(image, lang='rus+eng')
            
            if not text.strip():
                raise Exception("Не удалось распознать текст на изображении")
            
            return text.strip()
            
        except Exception as e:
            raise Exception(f"Ошибка OCR: {str(e)}")
    
    @staticmethod
    def _enhance_image_for_ocr(image: Image.Image) -> Image.Image:
        """Улучшение изображения для лучшего OCR"""
        
        # Масштабирование для улучшения качества
        width, height = image.size
        if width < 1500:
            scale_factor = 1500 / width
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Повышение контрастности
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Повышение резкости
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        return image
    
    @staticmethod
    def _extract_from_text(content: bytes) -> str:
        """Извлечение текста из текстового файла"""
        try:
            # Попытка декодирования в разных кодировках
            encodings = ['utf-8', 'utf-16', 'cp1251', 'latin1']
            
            for encoding in encodings:
                try:
                    text = content.decode(encoding)
                    return text.strip()
                except UnicodeDecodeError:
                    continue
            
            raise Exception("Не удалось декодировать текстовый файл")
            
        except Exception as e:
            raise Exception(f"Ошибка чтения текстового файла: {str(e)}")

# ============================================================================
# AI АНАЛИЗАТОР ДОКУМЕНТОВ
# ============================================================================

class DocumentAnalyzer:
    """Продвинутый анализатор юридических документов"""
    
    # Паттерны для поиска рисков
    RISK_PATTERNS = {
        "штрафные_санкции": {
            "keywords": ["штраф", "пеня", "неустойка", "санкции", "пени", "взыскание"],
            "severity": "высокий",
            "description": "Штрафные санкции и финансовые наказания"
        },
        "односторонние_условия": {
            "keywords": ["односторонн", "в любое время", "по своему усмотрению", "вправе расторгнуть", "без согласования"],
            "severity": "высокий", 
            "description": "Односторонние условия расторжения или изменения"
        },
        "высокая_ответственность": {
            "keywords": ["полная ответственность", "возмещение всех", "солидарная ответственность", "неограниченная ответственность"],
            "severity": "высокий",
            "description": "Повышенная финансовая ответственность"
        },
        "неопределенные_сроки": {
            "keywords": ["разумный срок", "в кратчайшие сроки", "незамедлительно", "в срочном порядке", "немедленно"],
            "severity": "средний",
            "description": "Неопределенные временные рамки"
        },
        "финансовые_риски": {
            "keywords": ["за свой счет", "без возмещения", "безвозмездно", "убытки покупателя", "собственные средства"],
            "severity": "средний",
            "description": "Дополнительные финансовые обязательства"
        },
        "форс_мажор": {
            "keywords": ["форс-мажор", "непреодолимая сила", "чрезвычайные обстоятельства"],
            "severity": "средний",
            "description": "Условия форс-мажора и ответственности"
        }
    }
    
    # Паттерны для поиска выгод
    BENEFIT_PATTERNS = {
        "гарантии": {
            "keywords": ["гарантия", "гарантирует", "обязуется", "гарантированно"],
            "value": "высокая",
            "description": "Гарантийные обязательства"
        },
        "защита_интересов": {
            "keywords": ["страхование", "компенсация", "возмещение ущерба", "защита интересов"],
            "value": "высокая",
            "description": "Защита интересов и компенсации"
        },
        "гибкие_условия": {
            "keywords": ["по согласованию", "возможность изменения", "право выбора", "по договоренности"],
            "value": "средняя",
            "description": "Гибкие условия сотрудничества"
        },
        "возврат_средств": {
            "keywords": ["возврат", "возмещение платежа", "компенсация расходов", "возврат денег"],
            "value": "высокая",
            "description": "Возможность возврата средств"
        },
        "дополнительные_услуги": {
            "keywords": ["бесплатно", "дополнительно", "в подарок", "без доплаты"],
            "value": "средняя",
            "description": "Дополнительные бесплатные услуги"
        }
    }
    
    # Неясные термины
    UNCLEAR_PATTERNS = [
        "в установленном порядке",
        "соответствующие меры", 
        "надлежащим образом",
        "разумный срок",
        "существенные нарушения",
        "форс-мажорные обстоятельства",
        "иные обстоятельства",
        "по обоюдному согласию",
        "в исключительных случаях",
        "в необходимых случаях",
        "при необходимости",
        "в зависимости от обстоятельств"
    ]
    
    @staticmethod
    def analyze(text: str, filename: str, file_type: str, file_size: int, word_count: int) -> DocumentAnalysis:
        """Комплексный анализ документа"""
        
        start_time = datetime.now()
        
        try:
            # Предобработка текста
            text_lower = text.lower()
            
            # Анализ рисков
            risks = DocumentAnalyzer._analyze_risks(text, text_lower)
            
            # Анализ выгод
            benefits = DocumentAnalyzer._analyze_benefits(text, text_lower)
            
            # Анализ неясных терминов
            unclear_terms = DocumentAnalyzer._analyze_unclear_terms(text, text_lower)
            
            # Создание подсветки
            highlights = DocumentAnalyzer._create_highlights(risks, benefits, unclear_terms)
            
            # Вычисление общей оценки
            overall_rating, risk_score = DocumentAnalyzer._calculate_rating(risks, benefits, unclear_terms)
            
            # Генерация рекомендаций
            recommendations = DocumentAnalyzer._generate_recommendations(risks, benefits, unclear_terms)
            
            # Создание резюме
            summary = DocumentAnalyzer._create_summary(filename, risks, benefits, unclear_terms, overall_rating)
            
            # Время обработки
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return DocumentAnalysis(
                document_id=str(uuid.uuid4()),
                filename=filename,
                file_type=file_type,
                file_size=file_size,
                text_content=text[:2000] + "..." if len(text) > 2000 else text,
                full_text=text,
                word_count=word_count,
                risks=risks,
                benefits=benefits,
                unclear_terms=unclear_terms,
                overall_rating=overall_rating,
                risk_score=risk_score,
                summary=summary,
                recommendations=recommendations,
                highlights=highlights,
                processed_at=datetime.now().isoformat(),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Ошибка анализа документа: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка анализа: {str(e)}")
    
    @staticmethod
    def _analyze_risks(text: str, text_lower: str) -> List[RiskData]:
        """Анализ рисков в документе"""
        risks = []
        
        for risk_type, pattern_data in DocumentAnalyzer.RISK_PATTERNS.items():
            for keyword in pattern_data["keywords"]:
                positions = DocumentAnalyzer._find_all_positions(text_lower, keyword)
                
                for pos in positions:
                    context = DocumentAnalyzer._extract_context(text, pos, len(keyword))
                    
                    risk = RiskData(
                        type=risk_type.replace("_", " ").title(),
                        keyword=keyword,
                        context=context,
                        position=pos,
                        severity=pattern_data["severity"],
                        recommendation=DocumentAnalyzer._get_risk_recommendation(risk_type, keyword),
                        legal_reference=DocumentAnalyzer._get_legal_reference(risk_type)
                    )
                    
                    risks.append(risk)
                    # Один риск на тип для избежания дублирования
                    break
                
                if risks and risks[-1].type == risk_type.replace("_", " ").title():
                    break
        
        return risks
    
    @staticmethod
    def _analyze_benefits(text: str, text_lower: str) -> List[BenefitData]:
        """Анализ выгодных условий"""
        benefits = []
        
        for benefit_type, pattern_data in DocumentAnalyzer.BENEFIT_PATTERNS.items():
            for keyword in pattern_data["keywords"]:
                positions = DocumentAnalyzer._find_all_positions(text_lower, keyword)
                
                for pos in positions:
                    context = DocumentAnalyzer._extract_context(text, pos, len(keyword))
                    
                    benefit = BenefitData(
                        type=benefit_type.replace("_", " ").title(),
                        keyword=keyword,
                        context=context,
                        position=pos,
                        value=pattern_data["value"],
                        description=pattern_data["description"]
                    )
                    
                    benefits.append(benefit)
                    break
                
                if benefits and benefits[-1].type == benefit_type.replace("_", " ").title():
                    break
        
        return benefits
    
    @staticmethod
    def _analyze_unclear_terms(text: str, text_lower: str) -> List[UnclearTermData]:
        """Анализ неясных терминов"""
        unclear_terms = []
        
        for phrase in DocumentAnalyzer.UNCLEAR_PATTERNS:
            positions = DocumentAnalyzer._find_all_positions(text_lower, phrase)
            
            for pos in positions:
                context = DocumentAnalyzer._extract_context(text, pos, len(phrase))
                
                unclear_term = UnclearTermData(
                    phrase=phrase,
                    context=context,
                    position=pos,
                    explanation=f"Фраза '{phrase}' требует конкретизации",
                    suggestion=DocumentAnalyzer._get_unclear_suggestion(phrase),
                    legal_clarification=DocumentAnalyzer._get_legal_clarification(phrase)
                )
                
                unclear_terms.append(unclear_term)
                break  # Один термин на фразу
        
        return unclear_terms
    
    @staticmethod
    def _find_all_positions(text: str, keyword: str) -> List[int]:
        """Поиск всех позиций ключевого слова"""
        positions = []
        start = 0
        
        while True:
            pos = text.find(keyword, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        
        return positions[:3]  # Максимум 3 вхождения
    
    @staticmethod
    def _extract_context(text: str, position: int, keyword_length: int, context_size: int = 150) -> str:
        """Извлечение контекста вокруг найденного слова"""
        start = max(0, position - context_size)
        end = min(len(text), position + keyword_length + context_size)
        
        context = text[start:end].strip()
        
        # Добавляем троеточие если контекст обрезан
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."
        
        return context
    
    @staticmethod
    def _create_highlights(risks: List[RiskData], benefits: List[BenefitData], unclear_terms: List[UnclearTermData]) -> List[HighlightData]:
        """Создание данных для подсветки"""
        highlights = []
        
        # Подсветка рисков
        for risk in risks:
            highlight = HighlightData(
                start=risk.position,
                end=risk.position + len(risk.keyword),
                type="risk",
                text=risk.keyword,
                category=risk.type,
                severity=risk.severity
            )
            highlights.append(highlight)
        
        # Подсветка выгод
        for benefit in benefits:
            highlight = HighlightData(
                start=benefit.position,
                end=benefit.position + len(benefit.keyword),
                type="benefit",
                text=benefit.keyword,
                category=benefit.type
            )
            highlights.append(highlight)
        
        # Подсветка неясных терминов
        for unclear in unclear_terms:
            highlight = HighlightData(
                start=unclear.position,
                end=unclear.position + len(unclear.phrase),
                type="unclear",
                text=unclear.phrase,
                category="Неясная формулировка"
            )
            highlights.append(highlight)
        
        # Сортировка по позиции
        highlights.sort(key=lambda x: x.start)
        
        return highlights
    
    @staticmethod
    def _calculate_rating(risks: List[RiskData], benefits: List[BenefitData], unclear_terms: List[UnclearTermData]) -> tuple:
        """Вычисление общей оценки документа"""
        
        # Подсчет рисков по уровням
        high_risks = len([r for r in risks if r.severity == "высокий"])
        medium_risks = len([r for r in risks if r.severity == "средний"]) 
        low_risks = len([r for r in risks if r.severity == "низкий"])
        
        # Подсчет выгод
        high_benefits = len([b for b in benefits if b.value == "высокая"])
        medium_benefits = len([b for b in benefits if b.value == "средняя"])
        
        # Расчет числовой оценки риска (0-100)
        risk_score = (high_risks * 30 + medium_risks * 15 + low_risks * 5 + len(unclear_terms) * 3)
        risk_score = min(100, risk_score)  # Максимум 100
        
        # Корректировка на выгоды
        benefit_adjustment = (high_benefits * 10 + medium_benefits * 5)
        risk_score = max(0, risk_score - benefit_adjustment)
        
        # Определение текстовой оценки
        if risk_score >= 60:
            overall_rating = "рискован"
        elif risk_score >= 30:
            overall_rating = "требует внимания"
        elif risk_score >= 10:
            overall_rating = "относительно безопасен"
        else:
            overall_rating = "безопасен"
        
        return overall_rating, risk_score
    
    @staticmethod
    def _generate_recommendations(risks: List[RiskData], benefits: List[BenefitData], unclear_terms: List[UnclearTermData]) -> List[str]:
        """Генерация рекомендаций"""
        recommendations = []
        
        if len(risks) > 0:
            recommendations.append("🔍 Внимательно изучите найденные риски перед подписанием")
            
        if any(r.severity == "высокий" for r in risks):
            recommendations.append("⚠️ Обнаружены высокие риски - рекомендуется консультация с юристом")
            
        if len(unclear_terms) > 0:
            recommendations.append("❓ Уточните неопределенные формулировки у контрагента")
            
        if len(benefits) > 0:
            recommendations.append("✅ Используйте найденные выгодные условия в переговорах")
            
        if len(risks) > 5:
            recommendations.append("📋 Рассмотрите возможность внесения изменений в договор")
            
        if not recommendations:
            recommendations.append("✨ Документ выглядит достаточно сбалансированным")
            
        return recommendations
    
    @staticmethod
    def _create_summary(filename: str, risks: List[RiskData], benefits: List[BenefitData], unclear_terms: List[UnclearTermData], overall_rating: str) -> str:
        """Создание краткого резюме"""
        
        high_risks = len([r for r in risks if r.severity == "высокий"])
        medium_risks = len([r for r in risks if r.severity == "средний"])
        
        summary = f"""📋 АНАЛИЗ ДОКУМЕНТА: {filename}

🎯 ОБЩАЯ ОЦЕНКА: {overall_rating.upper()}

📊 СТАТИСТИКА:
• Всего рисков: {len(risks)}
  - Высокие: {high_risks}
  - Средние: {medium_risks}
• Выгодные условия: {len(benefits)}
• Неясные формулировки: {len(unclear_terms)}

💡 КЛЮЧЕВЫЕ МОМЕНТЫ:
{DocumentAnalyzer._get_key_points(risks, benefits, unclear_terms)}

⚖️ РЕКОМЕНДАЦИЯ:
{DocumentAnalyzer._get_main_recommendation(overall_rating, high_risks)}

📋 Данный анализ основан на автоматическом поиске ключевых слов и фраз. 
Для окончательного решения рекомендуется консультация с юристом."""
        
        return summary.strip()
    
    @staticmethod
    def _get_key_points(risks: List[RiskData], benefits: List[BenefitData], unclear_terms: List[UnclearTermData]) -> str:
        """Получение ключевых моментов"""
        points = []
        
        if risks:
            main_risk = max(risks, key=lambda r: {"высокий": 3, "средний": 2, "низкий": 1}[r.severity])
            points.append(f"⚠️ Основной риск: {main_risk.type}")
            
        if benefits:
            main_benefit = benefits[0]
            points.append(f"✅ Главная выгода: {main_benefit.type}")
            
        if unclear_terms:
            points.append(f"❓ Требует уточнения: {unclear_terms[0].phrase}")
            
        return "\n".join(points) if points else "• Стандартные условия договора"
    
    @staticmethod
    def _get_main_recommendation(overall_rating: str, high_risks: int) -> str:
        """Получение основной рекомендации"""
        if overall_rating == "рискован":
            return "НЕ рекомендуется подписывать без внесения изменений"
        elif overall_rating == "требует внимания":
            return "Можно подписывать с осторожностью после изучения рисков"
        elif high_risks > 0:
            return "Рекомендуется консультация с юристом"
        else:
            return "Можно подписывать после ознакомления с условиями"
    
    @staticmethod
    def _get_risk_recommendation(risk_type: str, keyword: str) -> str:
        """Получение рекомендации по конкретному риску"""
        recommendations = {
            "штрафные_санкции": f"Изучите размеры и условия применения штрафных санкций за '{keyword}'",
            "односторонние_условия": f"Требуйте взаимности в условиях '{keyword}'",
            "высокая_ответственность": f"Рассмотрите ограничение ответственности по '{keyword}'",
            "неопределенные_сроки": f"Потребуйте конкретизации сроков вместо '{keyword}'",
            "финансовые_риски": f"Уточните финансовые обязательства по '{keyword}'",
            "форс_мажор": f"Изучите условия форс-мажора '{keyword}'"
        }
        return recommendations.get(risk_type, f"Обратите внимание на условие '{keyword}'")
    
    @staticmethod
    def _get_legal_reference(risk_type: str) -> Optional[str]:
        """Получение ссылки на законодательство РК"""
        references = {
            "штрафные_санкции": "Гражданский кодекс РК, статьи 350-354",
            "односторонние_условия": "Гражданский кодекс РК, статья 401",
            "высокая_ответственность": "Гражданский кодекс РК, статьи 359-364"
        }
        return references.get(risk_type)
    
    @staticmethod
    def _get_unclear_suggestion(phrase: str) -> str:
        """Получение предложения по неясному термину"""
        suggestions = {
            "разумный срок": "Укажите конкретные даты или количество дней",
            "в установленном порядке": "Опишите конкретную процедуру",
            "существенные нарушения": "Дайте определение существенности",
            "форс-мажорные обстоятельства": "Приведите исчерпывающий список"
        }
        return suggestions.get(phrase, f"Конкретизируйте понятие '{phrase}'")
    
    @staticmethod
    def _get_legal_clarification(phrase: str) -> Optional[str]:
        """Получение юридического разъяснения"""
        clarifications = {
            "разумный срок": "По ГК РК разумным считается срок, необходимый для выполнения обязательства",
            "существенные нарушения": "Нарушения, которые лишают сторону того, на что она рассчитывала"
        }
        return clarifications.get(phrase)

# ============================================================================
# СИСТЕМА СОХРАНЕНИЯ И УПРАВЛЕНИЯ
# ============================================================================

class DocumentManager:
    """Управление документами и результатами анализа"""
    
    def __init__(self):
        self.storage_dir = Path("storage")
        self.analysis_dir = self.storage_dir / "analysis"
        self.uploads_dir = self.storage_dir / "uploads"
        
        # Создание папок
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
    
    def save_analysis(self, analysis: DocumentAnalysis) -> str:
        """Сохранение результата анализа"""
        try:
            file_path = self.analysis_dir / f"{analysis.document_id}.json"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(analysis.dict(), f, ensure_ascii=False, indent=2)
            
            logger.info(f"Анализ сохранен: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Ошибка сохранения анализа: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка сохранения: {str(e)}")
    
    def load_analysis(self, document_id: str) -> Optional[DocumentAnalysis]:
        """Загрузка сохраненного анализа"""
        try:
            file_path = self.analysis_dir / f"{document_id}.json"
            
            if not file_path.exists():
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return DocumentAnalysis(**data)
            
        except Exception as e:
            logger.error(f"Ошибка загрузки анализа {document_id}: {e}")
            return None
    
    def get_saved_analyses(self) -> List[Dict]:
        """Получение списка сохраненных анализов"""
        try:
            analyses = []
            
            for file_path in self.analysis_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    analyses.append({
                        "document_id": data.get("document_id"),
                        "filename": data.get("filename"),
                        "overall_rating": data.get("overall_rating"),
                        "risk_score": data.get("risk_score"),
                        "processed_at": data.get("processed_at"),
                        "file_size": data.get("file_size"),
                        "word_count": data.get("word_count")
                    })
                except Exception as e:
                    logger.warning(f"Ошибка чтения файла {file_path}: {e}")
                    continue
            
            # Сортировка по дате обработки (новые первыми)
            analyses.sort(key=lambda x: x.get("processed_at", ""), reverse=True)
            
            return analyses
            
        except Exception as e:
            logger.error(f"Ошибка получения списка анализов: {e}")
            return []

# ============================================================================
# WEB ПРИЛОЖЕНИЕ
# ============================================================================

# Инициализация FastAPI
app = FastAPI(
    title="🛡️ SafeDocs - AI защита юридических документов",
    description="Профессиональный анализ договоров с фокусом на казахстанское законодательство",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация компонентов
document_manager = DocumentManager()

# Временное хранилище для сессий (в продакшене - Redis)
session_storage = {}

# Подключение статических файлов
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.warning(f"Не удалось подключить статические файлы: {e}")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def get_main_page():
    """Главная страница приложения"""
    try:
        return FileResponse('static/index.html')
    except FileNotFoundError:
        return JSONResponse(
            status_code=404,
            content={
                "message": "Главная страница не найдена",
                "error": "Создайте файл static/index.html",
                "api_docs": "/api/docs"
            }
        )

@app.post("/api/analyze", response_model=DocumentAnalysis)
async def analyze_document(file: UploadFile = File(...)):
    """Анализ загруженного документа"""
    
    logger.info(f"Начат анализ файла: {file.filename}")
    
    try:
        # Обработка файла
        file_data = await FileProcessor.process_file(file)
        
        # Анализ документа
        analysis = DocumentAnalyzer.analyze(
            text=file_data['text'],
            filename=file_data['filename'],
            file_type=file_data['file_type'],
            file_size=file_data['file_size'],
            word_count=file_data['word_count']
        )
        
        # Сохранение в сессионное хранилище
        session_storage[analysis.document_id] = analysis
        
        logger.info(f"Анализ завершен: {analysis.document_id}")
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка при анализе: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")

@app.post("/api/save")
async def save_analysis(request: dict):
    """Сохранение результата анализа"""
    
    document_id = request.get("document_id")
    
    if not document_id:
        raise HTTPException(status_code=400, detail="ID документа не указан")
    
    if document_id not in session_storage:
        raise HTTPException(status_code=404, detail="Документ не найден в сессии")
    
    try:
        analysis = session_storage[document_id]
        file_path = document_manager.save_analysis(analysis)
        
        return {
            "message": "Анализ успешно сохранен",
            "document_id": document_id,
            "file_path": file_path,
            "saved_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Ошибка сохранения анализа {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка сохранения: {str(e)}")

@app.get("/api/saved")
async def get_saved_documents():
    """Получение списка сохраненных документов"""
    
    try:
        saved_docs = document_manager.get_saved_analyses()
        
        return {
            "saved_documents": saved_docs,
            "total_count": len(saved_docs)
        }
        
    except Exception as e:
        logger.error(f"Ошибка получения сохраненных документов: {e}")
        raise HTTPException(status_code=500, detail="Ошибка получения данных")

@app.get("/api/documents/{document_id}")
async def get_document_analysis(document_id: str):
    """Получение результата анализа по ID"""
    
    # Сначала ищем в сессионном хранилище
    if document_id in session_storage:
        return session_storage[document_id]
    
    # Затем в сохраненных файлах
    analysis = document_manager.load_analysis(document_id)
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Документ не найден")
    
    return analysis

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_document(message: ChatMessage):
    """Чат с AI по документу"""
    
    # Поиск документа
    analysis = None
    
    if message.document_id in session_storage:
        analysis = session_storage[message.document_id]
    else:
        analysis = document_manager.load_analysis(message.document_id)
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Документ не найден")
    
    try:
        # Простая система ответов (в будущем - интеграция с AI)
        answer = DocumentAnalyzer._generate_chat_response(message.question, analysis)
        
        return ChatResponse(
            answer=answer,
            confidence=0.8,
            relevant_sections=[],
            document_id=message.document_id
        )
        
    except Exception as e:
        logger.error(f"Ошибка чата: {e}")
        raise HTTPException(status_code=500, detail="Ошибка обработки запроса")

@app.get("/api/health")
async def health_check():
    """Проверка состояния системы"""
    
    return {
        "status": "healthy",
        "service": "SafeDocs",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "session_documents": len(session_storage),
        "saved_documents": len(document_manager.get_saved_analyses())
    }

@app.get("/api/stats")
async def get_statistics():
    """Статистика использования"""
    
    try:
        saved_docs = document_manager.get_saved_analyses()
        
        # Подсчет статистики
        total_docs = len(saved_docs)
        ratings_count = {}
        avg_risk_score = 0
        
        for doc in saved_docs:
            rating = doc.get("overall_rating", "неизвестно")
            ratings_count[rating] = ratings_count.get(rating, 0) + 1
            avg_risk_score += doc.get("risk_score", 0)
        
        if total_docs > 0:
            avg_risk_score = avg_risk_score / total_docs
        
        return {
            "total_documents": total_docs,
            "session_documents": len(session_storage),
            "ratings_distribution": ratings_count,
            "average_risk_score": round(avg_risk_score, 2),
            "service_uptime": "running"
        }
        
    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}")
        raise HTTPException(status_code=500, detail="Ошибка получения статистики")

# Расширение DocumentAnalyzer для чата
@staticmethod
def _generate_chat_response(question: str, analysis: DocumentAnalysis) -> str:
    """Генерация ответа в чате"""
    
    question_lower = question.lower()
    
    # Анализ типа вопроса и генерация ответа
    if any(word in question_lower for word in ["риск", "опасн", "штраф", "проблем"]):
        if analysis.risks:
            risks_text = "\n".join([f"• {r.type}: {r.recommendation}" for r in analysis.risks[:3]])
            return f"🚨 В документе обнаружены следующие риски:\n\n{risks_text}\n\nОбщий уровень риска: {analysis.risk_score}/100"
        else:
            return "✅ Серьезных рисков в документе не обнаружено."
    
    elif any(word in question_lower for word in ["выгод", "плюс", "хорош", "положительн"]):
        if analysis.benefits:
            benefits_text = "\n".join([f"• {b.type}: {b.description}" for b in analysis.benefits[:3]])
            return f"✅ Выгодные условия в документе:\n\n{benefits_text}"
        else:
            return "❌ Явных выгодных условий не обнаружено."
    
    elif any(word in question_lower for word in ["непонятн", "неясн", "что означает", "расшифр"]):
        if analysis.unclear_terms:
            unclear_text = "\n".join([f"• {u.phrase}: {u.suggestion}" for u in analysis.unclear_terms[:3]])
            return f"❓ Неясные формулировки в документе:\n\n{unclear_text}"
        else:
            return "✅ Все основные формулировки достаточно понятны."
    
    elif any(word in question_lower for word in ["подписывать", "согласиться", "стоит ли", "рекоменд"]):
        recommendations_text = "\n".join([f"• {rec}" for rec in analysis.recommendations])
        return f"""
🤔 Рекомендация по документу "{analysis.filename}":

📊 Общая оценка: {analysis.overall_rating}
📈 Уровень риска: {analysis.risk_score}/100

💡 Рекомендации:
{recommendations_text}

⚖️ Окончательное решение остается за вами. При наличии сомнений рекомендуется консультация с юристом.
        """.strip()
    
    elif any(word in question_lower for word in ["резюме", "итог", "кратко", "суть"]):
        return analysis.summary
    
    else:
        return f"""
💬 Спасибо за вопрос! 

Я могу помочь с анализом документа "{analysis.filename}":

🔍 Доступная информация:
• Найдено рисков: {len(analysis.risks)}
• Выгодных условий: {len(analysis.benefits)}
• Неясных формулировок: {len(analysis.unclear_terms)}
• Общая оценка: {analysis.overall_rating}

❓ Попробуйте спросить:
• "Какие есть риски?"
• "Есть ли выгодные условия?"
• "Стоит ли подписывать?"
• "Что означают неясные термины?"

🤖 Система постоянно улучшается для предоставления более точных ответов.
        """

# Добавление метода в класс
DocumentAnalyzer._generate_chat_response = _generate_chat_response

# ============================================================================
# ЗАПУСК ПРИЛОЖЕНИЯ
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🛡️  SafeDocs v2.0 - AI защита юридических документов")
    print("=" * 60)
    print("🌐 Веб-приложение: http://localhost:8000")
    print("📚 API документация: http://localhost:8000/api/docs")
    print("🔬 Тестирование API: http://localhost:8000/api/redoc")
    print("💾 Данные сохраняются в: ./storage/")
    print("=" * 60)
    
    # Создание необходимых папок
    Path("static").mkdir(exist_ok=True)
    Path("storage").mkdir(exist_ok=True)
    Path("storage/analysis").mkdir(exist_ok=True)
    Path("storage/uploads").mkdir(exist_ok=True)
    
    
    
    
    # Запуск сервера
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
    
    
  