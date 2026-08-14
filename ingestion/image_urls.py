from dataclasses import dataclass


CDN_TEMPLATE = "https://m.media-amazon.com/images/I/{image_id}.jpg"


@dataclass
class Imageresult:
    main_url: str | None
    other_urls: list[str]
    used_fallback: bool
    has_no_image: bool

def build_image_url(image_id:str):
    return CDN_TEMPLATE.format(image_id=image_id)

def extract_images(item:dict)->Imageresult:
    main_id=item.get("main_image_id")
    other_ids=item.get("other_image_id")
    other_urls = [build_image_url(i) for i in other_ids]

    if main_id:
        return Imageresult(
            main_url=build_image_url(main_id),
            other_urls=other_urls,
            used_fallback=False,
            has_no_image=False,
        )
    if other_urls:
        return ImageResult(
            main_url=other_urls[0],
            other_urls=other_urls[1:],
            used_fallback=True,
            has_no_image=False,
        )
    return ImageResult(main_url=None,other_urls=[],used_fallback=False,has_no_image=True)
    

