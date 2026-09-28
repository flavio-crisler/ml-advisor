import {
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import { useTranslation } from "react-i18next";

type TooltipPosition = {
  left: number;
  top: number;
  below: boolean;
};

/**
 * Componente de tooltip de ajuda acessível com posicionamento dinâmico e suporte a teclado e toque.
 *
 * @param props.content - Conteúdo informativo a exibir no balão flutuante.
 * @param props.label - Rótulo acessível opcional do elemento referenciado.
 * @returns Elemento com botão de gatilho e portal de tooltip flutuante.
 */
export function InfoTooltip({
  content,
  label,
}: {
  content: ReactNode;
  label?: string;
}) {
  const { t } = useTranslation("common");
  const tooltipId = useId();
  const triggerRef = useRef<HTMLButtonElement>(null);
  const [hovered, setHovered] = useState(false);
  const [focused, setFocused] = useState(false);
  const [pinned, setPinned] = useState(false);
  const [dismissedWhileFocused, setDismissedWhileFocused] = useState(false);
  const [position, setPosition] = useState<TooltipPosition | null>(null);
  const open = hovered || (focused && !dismissedWhileFocused) || pinned;
  const ariaLabel = label
    ? t("help.moreInformationFor", { label })
    : t("help.moreInformation");

  useLayoutEffect(() => {
    if (!open) return;
    const updatePosition = () => {
      const trigger = triggerRef.current;
      if (!trigger) return;
      const bounds = trigger.getBoundingClientRect();
      const halfWidth = Math.min(144, Math.max(0, window.innerWidth / 2 - 16));
      setPosition({
        left: Math.min(
          window.innerWidth - halfWidth,
          Math.max(halfWidth, bounds.left + bounds.width / 2),
        ),
        top: bounds.top > 140 ? bounds.top - 8 : bounds.bottom + 8,
        below: bounds.top <= 140,
      });
    };
    updatePosition();
    window.addEventListener("resize", updatePosition);
    window.addEventListener("scroll", updatePosition, true);
    return () => {
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const dismiss = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setHovered(false);
      setPinned(false);
      setDismissedWhileFocused(true);
    };
    document.addEventListener("keydown", dismiss);
    return () => document.removeEventListener("keydown", dismiss);
  }, [open]);

  useEffect(() => {
    if (!pinned) return;
    const dismiss = (event: PointerEvent) => {
      if (triggerRef.current?.contains(event.target as Node)) return;
      setPinned(false);
    };
    document.addEventListener("pointerdown", dismiss);
    return () => document.removeEventListener("pointerdown", dismiss);
  }, [pinned]);

  return (
    <span className="inline-flex align-middle">
      <button
        ref={triggerRef}
        type="button"
        className="inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-muted transition-colors hover:bg-primary/10 hover:text-primary"
        aria-label={ariaLabel}
        aria-expanded={open}
        aria-describedby={open ? tooltipId : undefined}
        onMouseEnter={() => {
          setHovered(true);
          setDismissedWhileFocused(false);
        }}
        onMouseLeave={() => setHovered(false)}
        onFocus={() => {
          setFocused(true);
          setDismissedWhileFocused(false);
        }}
        onBlur={() => {
          setFocused(false);
          setDismissedWhileFocused(false);
        }}
        onClick={(event) => {
          event.preventDefault();
          event.stopPropagation();
          if (pinned) {
            setPinned(false);
            setHovered(false);
            setDismissedWhileFocused(true);
          } else {
            setPinned(true);
            setDismissedWhileFocused(false);
          }
        }}
      >
        <svg
          aria-hidden="true"
          viewBox="0 0 20 20"
          className="h-4 w-4"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.7"
        >
          <circle cx="10" cy="10" r="7.5" />
          <path strokeLinecap="round" d="M10 9v5M10 6.2h.01" />
        </svg>
      </button>
      {open && position
        ? createPortal(
            <span
              id={tooltipId}
              role="tooltip"
              data-state="open"
              className="pointer-events-none fixed z-[100] w-[min(18rem,calc(100vw-2rem))] rounded-lg bg-code px-3 py-2 text-left text-xs font-normal leading-5 text-code-text shadow-xl"
              style={{
                left: position.left,
                top: position.top,
                transform: position.below
                  ? "translate(-50%, 0)"
                  : "translate(-50%, -100%)",
              }}
            >
              {content}
            </span>,
            document.body,
          )
        : null}
    </span>
  );
}

/**
 * Rótulo com ícone de tooltip de ajuda integrado.
 *
 * @param props.children - Texto ou elemento filho do rótulo.
 * @param props.help - Conteúdo de ajuda a ser exibido no tooltip associado.
 * @returns Elemento span contendo o texto e o tooltip de ajuda.
 */
export function HelpLabel({
  children,
  help,
}: {
  children: ReactNode;
  help: ReactNode;
}) {
  const label = typeof children === "string" ? children : undefined;
  return (
    <span className="inline-flex items-center gap-1.5">
      <span>{children}</span>
      <InfoTooltip content={help} label={label} />
    </span>
  );
}
