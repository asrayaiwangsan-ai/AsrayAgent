from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated, Optional, Literal
import os
import asyncio
from bs4 import BeautifulSoup, NavigableString
import base64
import re
import json

from langchain_core.messages import ToolMessage
base_dir = os.getenv("ASRAY_BASE_DIR", "/home/ai/asraydata")

try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page
except ImportError:
    pass

try:
    import httpx
except ImportError:
    pass


class BrowserManager:
    _playwright = None
    _browser = None

    @classmethod
    async def get_browser(cls) -> Browser:
        if cls._playwright is None:
            cls._playwright = await async_playwright().start()
        if cls._browser is None or not cls._browser.is_connected():
            cls._browser = await cls._playwright.chromium.launch(headless=True)
        return cls._browser

    @classmethod
    async def close(cls):
        if cls._browser:
            await cls._browser.close()
        if cls._playwright:
            await cls._playwright.stop()


_INTERACTIVE_TAGS = {"a", "button", "input", "textarea", "select"}
_USELESS_TAGS = {"script", "style", "meta", "link", "noscript", "svg", "path", "iframe", "canvas"}


def _is_interactive(el) -> bool:
    tag = el.name
    if tag in _INTERACTIVE_TAGS:
        return True
    if el.get("role") in ("button", "link", "textbox", "combobox", "listbox", "menuitem", "option", "tab", "switch", "checkbox", "radio"):
        return True
    if el.has_attr("onclick"):
        return True
    if el.has_attr("tabindex") and tag in ("div", "span", "li", "tr", "td"):
        return True
    return False


def _build_selector(el) -> str:
    tag = el.name

    if el.get("id"):
        return f"#{el['id']}"

    if el.get("data-testid"):
        return f"[data-testid='{el['data-testid']}']"

    if el.get("name"):
        return f"{tag}[name='{el['name']}']"

    if tag == "input" and el.get("type"):
        return f"input[type='{el['type']}']"

    if el.get("aria-label"):
        lbl = el["aria-label"][:50]
        return f"[aria-label='{lbl}']"

    if el.get("role"):
        return f"[role='{el['role']}']"

    if el.get("class"):
        classes = el.get("class")
        if isinstance(classes, list):
            cls = ".".join(classes[:2])
        else:
            cls = classes.replace(" ", ".")[:40]
        return f"{tag}.{cls}"

    if el.get("title"):
        t = el["title"][:50]
        return f"{tag}[title='{t}']"

    return tag


def _clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for useless in soup(_USELESS_TAGS):
        useless.decompose()

    all_interactive = [el for el in soup.find_all() if _is_interactive(el)]
    all_interactive_set = set(all_interactive)

    leaf_interactive = []
    for el in all_interactive:
        has_interactive_descendant = any(
            d in all_interactive_set for d in el.descendants
            if isinstance(d, type(el))
        )
        if not has_interactive_descendant:
            leaf_interactive.append(el)

    for el in leaf_interactive:
        selector = _build_selector(el)

        if el.name == "select":
            options = []
            for opt in el.find_all("option"):
                val = opt.get("value", "")
                txt = opt.get_text(strip=True)[:20]
                options.append(f"{txt}={val}" if val else txt)
            label = " | ".join(options[:15])
            inject = f" [SELECT: {label} | selector: {selector}] "
            el.replace_with(inject)
            continue

        label = ""
        if el.name == "input":
            label = el.get("placeholder") or el.get("value") or el.get("aria-label") or el.get("title") or el.get("type", "")
        elif el.name == "textarea":
            label = el.get("placeholder") or el.get("aria-label") or ""
        else:
            label = el.get_text(strip=True)

        if not label:
            label = el.get("title") or el.get("aria-label") or ""

        tag_desc = el.name.upper()
        if el.name not in _INTERACTIVE_TAGS:
            tag_desc = el.get("role", "BTN").upper()

        inject = f" [{tag_desc}: {label.strip()[:30]} | selector: {selector}] "
        el.replace_with(inject)

    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r'\s+', ' ', text)
    return text


_PAGE_CACHE = {}

@tool
async def browser_tool(
    action: Literal["navigate", "click", "type", "extract", "screenshot", "scroll", "press"],
    url: Optional[str] = None,
    selector: Optional[str] = None,
    text: Optional[str] = None,
    wait_ms: int = 1000,
    extra: dict = {},
    multimodal: Annotated[bool, InjectedToolArg()] = False ,
    tool_call_id: Annotated[str, InjectedToolArg()] =""
) -> str:
    """
    一个强大的浏览器工具，可以执行导航、点击、输入、内容提取和截图。
    跨调用保持浏览器状态（Cookie/Session）。

    :param action: 要执行的操作。
    :param url: 导航目标 URL (仅在 navigate 时需要)。
    :param selector: CSS 选择器 (在 click, type, scroll 时需要)。
    :param text: 要输入的文本 (在 type 和 press 时需要)。
    :param wait_ms: 操作后的等待时间（毫秒）。
    """

    session_key = extra.get("thread_id", "default_browser_session")
    page: Page = _PAGE_CACHE.get(session_key)

    try:
        if page is None or page.is_closed():
            browser = await BrowserManager.get_browser()
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
            )
            page = await context.new_page()
            _PAGE_CACHE[session_key] = page
            extra["_has_browser_session"] = session_key

        if action == "navigate":
            if not url:
                return "错误：navigate 操作需要 url 参数。"
            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
            except Exception:
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=15000)
                except Exception:
                    await page.goto(url, wait_until="load", timeout=15000)

        elif action == "click":
            if not selector:
                return "错误：click 操作需要 selector 参数。"
            try:
                await page.click(selector, timeout=5000)
            except Exception:
                try:
                    await page.locator(selector).click(force=True, timeout=3000)
                except Exception:
                    await page.locator(selector).dispatch_event("click")

        elif action == "type":
            if not selector or text is None:
                return "错误：type 操作需要 selector 和 text 参数。"
            try:
                await page.fill(selector, text, timeout=5000)
            except Exception:
                safe_text = text.replace("'", "\\'")
                await page.evaluate(f"""
                    const el = document.querySelector('{selector}');
                    if (el) {{
                        el.value = '{safe_text}';
                        el.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    }}
                """)

        elif action == "press":
            if not text:
                return "错误：press 操作需要 text 参数（如 Enter、Escape、Tab）。"
            await page.keyboard.press(text)

        elif action == "scroll":
            if selector:
                await page.locator(selector).scroll_into_view_if_needed()
            else:
                await page.evaluate("window.scrollBy(0, window.innerHeight)")

        elif action == "screenshot":
            if not multimodal:
                return ToolMessage(
                    content=[{"type": "text", "text": f"只有多模态模型参能使用截图功能，如果你是多模态模型，请体系用户修改配置"}],
                    tool_call_id=tool_call_id
                )
            
            
            screenshot = await page.screenshot(full_page=False)
            import uuid
            from pathlib import Path
            
            save_path = base_dir + "/images/" + uuid.uuid4().__str__() + ".png"
            Path(save_path).write_bytes(screenshot)
            
            return ToolMessage(

                    content=[{"type": "image_ref", "image_ref": {"path": save_path}}],
                    tool_call_id=tool_call_id
            )
                            
                            


        if wait_ms > 0:
            await page.wait_for_timeout(wait_ms)

        content = await page.content()
        cleaned_text = _clean_html(content)

        limit = 20000
        if len(cleaned_text) > limit:
            cleaned_text = cleaned_text[:limit] + f"\n\n[... 内容过长，已截断 {len(cleaned_text) - limit} 字符 ...]"

        return f"操作 {action} 完成。当前页面标题: {await page.title()}\n内容摘要：\n{cleaned_text}"

    except Exception as e:
        return f"浏览器操作失败：{type(e).__name__}: {str(e)}"


@tool
async def web_fetch(
    url: str,
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = "GET",
    timeout: int = 30,
    headers: Optional[dict] = None,
    body: Optional[str] = None,
    max_length: int = 5000,
    user_agent: Optional[str] = None,
) -> dict:
    """
    通用 HTTP 请求工具。支持多种 HTTP 方法、JSON 请求体
    对于 HTML 页面自动清洗提取文本；对于 JSON API 返回结构化数据。

    :param url: 目标 URL
    :param method: HTTP 方法，默认 GET。POST/PUT/PATCH 时需传 body
    :param timeout: 请求超时时间（秒），SSE 模式下为流的总超时，默认 30 秒
    :param headers: 自定义请求头，会合并到默认头之上
    :param body: 请求体。JSON 字符串，如 '{"key":"value"}'
    :param max_length: 文本类响应内容的最大长度，默认 5000
    :param user_agent: 自定义 User-Agent

    :return: 包含 status、content_type、content 
    """

    default_headers = {
        "User-Agent": user_agent or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml,application/json;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }

    if headers:
        default_headers.update(headers)

    json_body = None
    if body and method in ("POST", "PUT", "PATCH"):
        try:
            json_body = json.loads(body)
        except json.JSONDecodeError:
            return {"status": None, "error": f"body 不是合法的 JSON 字符串: {body[:200]}"}

    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:


            req = client.build_request(method, url, headers=default_headers, json=json_body, timeout=timeout)
            response = await client.send(req)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")

            if "application/json" in content_type:
                return {
                    "status": response.status_code,
                    "url": str(response.url),
                    "content_type": "json",
                    "data": response.json(),
                }

            if "text/html" in content_type:
                soup = BeautifulSoup(response.text, "html.parser")
                title = ""
                title_tag = soup.find("title")
                if title_tag:
                    title = title_tag.get_text(strip=True)

                for useless in soup(_USELESS_TAGS):
                    useless.decompose()

                content = soup.get_text(separator="\n", strip=True)
                content = re.sub(r'\n\s*\n', '\n\n', content)

                if len(content) > max_length:
                    content = content[:max_length] + f"\n\n[... 内容已截断，超出 {max_length} 字符 ...]"

                return {
                    "status": response.status_code,
                    "url": str(response.url),
                    "content_type": "html",
                    "title": title,
                    "content": content,
                }

            text = response.text
            if len(text) > max_length:
                text = text[:max_length] + f"\n[... 截断 {len(text) - max_length} 字符]"

            return {
                "status": response.status_code,
                "url": str(response.url),
                "content_type": content_type or "text",
                "content": text,
            }

    except httpx.HTTPStatusError as e:
        return {
            "status": e.response.status_code if e.response else None,
            "error": f"HTTP {e.response.status_code}: {e.response.reason_phrase}",
            "response_body": e.response.text[:2000] if e.response else None,
        }
    except httpx.RequestError as e:
        return {"status": None, "error": f"请求错误：{type(e).__name__} - {str(e)}"}
    except Exception as e:
        return {"status": None, "error": f"未知错误：{type(e).__name__} - {str(e)}"}




browser_tools_map = {
    "browser_tool": browser_tool,
    "web_fetch": web_fetch,
}




