"""Mock data representing the video knowledge graph."""

from typing import Dict, List, Any

MOCK_DATA: Dict[str, List[Dict[str, Any]]] = {
    "narrative": [
        # Video A - Cooking
        {
            "id": 1,
            "text": "Chef aggressively smashes wagyu beef patty onto hot cast iron skillet, creating sizzle and smoke",
            "video_id": "video_A",
            "video_title": "Gordon Ramsay Style Burger",
            "creator_id": "chef_gordon_fan",
            "category": "cooking",
            "start_s": 10.0,
            "end_s": 20.0,
            "views": 1500000,
            "likes": 89000,
            "upload_date": "2025-01-15"
        },
        {
            "id": 2,
            "text": "Chef seasons patty from height letting salt crystals fall evenly across the surface",
            "video_id": "video_A",
            "video_title": "Gordon Ramsay Style Burger",
            "creator_id": "chef_gordon_fan",
            "category": "cooking",
            "start_s": 20.0,
            "end_s": 30.0,
            "views": 1500000,
            "likes": 89000,
            "upload_date": "2025-01-15"
        },
        # Video B - Tech
        {
            "id": 3,
            "text": "Reviewer struggles to fit massive graphics card into case, pushing hard against PCIe slot",
            "video_id": "video_B",
            "video_title": "RTX 5090 Unboxing",
            "creator_id": "tech_reviewer_99",
            "category": "tech",
            "start_s": 20.0,
            "end_s": 30.0,
            "views": 2300000,
            "likes": 156000,
            "upload_date": "2025-02-20"
        },
        {
            "id": 4,
            "text": "Reviewer carefully removes GPU from anti-static bag and inspects the triple fan cooler",
            "video_id": "video_B",
            "video_title": "RTX 5090 Unboxing",
            "creator_id": "tech_reviewer_99",
            "category": "tech",
            "start_s": 5.0,
            "end_s": 15.0,
            "views": 2300000,
            "likes": 156000,
            "upload_date": "2025-02-20"
        },
        # Video C - Fashion
        {
            "id": 5,
            "text": "Creator does slow 360 spin to show skirt flow and movement of fabric",
            "video_id": "video_C",
            "video_title": "Summer Haul 2025",
            "creator_id": "fashion_influencer",
            "category": "fashion",
            "start_s": 50.0,
            "end_s": 60.0,
            "views": 890000,
            "likes": 67000,
            "upload_date": "2025-03-10"
        },
        # Video D - Fitness
        {
            "id": 6,
            "text": "Trainer demonstrates proper squat form with barbell, keeping back straight and knees tracking over toes",
            "video_id": "video_D",
            "video_title": "Perfect Squat Form",
            "creator_id": "fitness_coach",
            "category": "fitness",
            "start_s": 15.0,
            "end_s": 25.0,
            "views": 3200000,
            "likes": 245000,
            "upload_date": "2025-02-28"
        },
    ],
    "transcript": [
        # Video A - Cooking
        {
            "id": 101,
            "text": "Always season from a height, this gives you even distribution across the entire patty",
            "video_id": "video_A",
            "video_title": "Gordon Ramsay Style Burger",
            "creator_id": "chef_gordon_fan",
            "category": "cooking",
            "start_s": 12.0,
            "end_s": 16.0,
            "speaker": "Chef",
            "views": 1500000,
            "likes": 89000,
            "upload_date": "2025-01-15"
        },
        {
            "id": 102,
            "text": "The key to a good smash burger is getting that cast iron absolutely screaming hot",
            "video_id": "video_A",
            "video_title": "Gordon Ramsay Style Burger",
            "creator_id": "chef_gordon_fan",
            "category": "cooking",
            "start_s": 5.0,
            "end_s": 9.0,
            "speaker": "Chef",
            "views": 1500000,
            "likes": 89000,
            "upload_date": "2025-01-15"
        },
        # Video B - Tech
        {
            "id": 103,
            "text": "This thing is absolutely massive, I'm genuinely worried it won't fit in my case",
            "video_id": "video_B",
            "video_title": "RTX 5090 Unboxing",
            "creator_id": "tech_reviewer_99",
            "category": "tech",
            "start_s": 25.0,
            "end_s": 29.0,
            "speaker": "Reviewer",
            "views": 2300000,
            "likes": 156000,
            "upload_date": "2025-02-20"
        },
        {
            "id": 104,
            "text": "128 gigabytes of VRAM, that's just insane for a consumer card",
            "video_id": "video_B",
            "video_title": "RTX 5090 Unboxing",
            "creator_id": "tech_reviewer_99",
            "category": "tech",
            "start_s": 8.0,
            "end_s": 12.0,
            "speaker": "Reviewer",
            "views": 2300000,
            "likes": 156000,
            "upload_date": "2025-02-20"
        },
        # Video C - Fashion
        {
            "id": 105,
            "text": "It feels a bit cheaper than I expected for a Zara piece at this price point",
            "video_id": "video_C",
            "video_title": "Summer Haul 2025",
            "creator_id": "fashion_influencer",
            "category": "fashion",
            "start_s": 55.0,
            "end_s": 59.0,
            "speaker": "Creator",
            "views": 890000,
            "likes": 67000,
            "upload_date": "2025-03-10"
        },
        # Video D - Fitness
        {
            "id": 106,
            "text": "Keep your core tight and your chest up, don't let those knees cave inward",
            "video_id": "video_D",
            "video_title": "Perfect Squat Form",
            "creator_id": "fitness_coach",
            "category": "fitness",
            "start_s": 18.0,
            "end_s": 22.0,
            "speaker": "Trainer",
            "views": 3200000,
            "likes": 245000,
            "upload_date": "2025-02-28"
        },
    ],
    "image": [
        # Video A - Cooking
        {
            "id": 201,
            "text": "Close-up of premium Wagyu beef label on black packaging with gold lettering",
            "video_id": "video_A",
            "video_title": "Gordon Ramsay Style Burger",
            "creator_id": "chef_gordon_fan",
            "category": "cooking",
            "start_s": 10.0,
            "end_s": 10.0,
            "visual_objects": ["Wagyu label", "beef packaging"],
            "ocr_text": "A5 Wagyu Premium",
            "views": 1500000,
            "likes": 89000,
            "upload_date": "2025-01-15"
        },
        # Video B - Tech
        {
            "id": 202,
            "text": "RTX 5090 retail box showing 128GB VRAM specification in large white text on green accent",
            "video_id": "video_B",
            "video_title": "RTX 5090 Unboxing",
            "creator_id": "tech_reviewer_99",
            "category": "tech",
            "start_s": 5.0,
            "end_s": 5.0,
            "visual_objects": ["RTX 5090 box", "NVIDIA logo"],
            "ocr_text": "128GB VRAM",
            "views": 2300000,
            "likes": 156000,
            "upload_date": "2025-02-20"
        },
        {
            "id": 203,
            "text": "Massive triple-fan GPU barely fitting into mid-tower case, cables strained",
            "video_id": "video_B",
            "video_title": "RTX 5090 Unboxing",
            "creator_id": "tech_reviewer_99",
            "category": "tech",
            "start_s": 25.0,
            "end_s": 25.0,
            "visual_objects": ["graphics card", "PC case", "cables"],
            "ocr_text": "",
            "views": 2300000,
            "likes": 156000,
            "upload_date": "2025-02-20"
        },
        # Video C - Fashion
        {
            "id": 204,
            "text": "Zara brand tag visible on cream-colored flowing midi skirt",
            "video_id": "video_C",
            "video_title": "Summer Haul 2025",
            "creator_id": "fashion_influencer",
            "category": "fashion",
            "start_s": 50.0,
            "end_s": 50.0,
            "visual_objects": ["Zara tag", "skirt", "clothing label"],
            "ocr_text": "ZARA",
            "views": 890000,
            "likes": 67000,
            "upload_date": "2025-03-10"
        },
        # Video D - Fitness
        {
            "id": 205,
            "text": "Side view of trainer in squat position with Olympic barbell, gym environment with mirrors",
            "video_id": "video_D",
            "video_title": "Perfect Squat Form",
            "creator_id": "fitness_coach",
            "category": "fitness",
            "start_s": 20.0,
            "end_s": 20.0,
            "visual_objects": ["barbell", "squat rack", "gym mirrors"],
            "ocr_text": "",
            "views": 3200000,
            "likes": 245000,
            "upload_date": "2025-02-28"
        },
    ]
}
