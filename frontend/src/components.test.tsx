import { afterEach, describe, expect, it, vi } from "vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { StatusNotice, ReviewEditor } from "./components";
import { ApiError } from "./api";
import type { Draft } from "./types";
afterEach(cleanup);
describe("обязательные состояния", () => {
  for (const state of [
    "loading",
    "empty",
    "error",
    "unsupported",
    "missing_reference",
    "analysis_failed",
  ] as const) {
    it(state, () => {
      render(<StatusNotice state={state} />);
      expect(
        screen.getByRole(
          state === "error" || state === "analysis_failed" ? "alert" : "status",
        ).textContent,
      ).toBeTruthy();
    });
  }
});
it("не принимает правку без подтверждения ошибки", () => {
  const draft: Draft = {
    assessment: "uncertain",
    decision: "deferred",
    comment: "",
    suggestionId: "s",
  };
  const onChange = vi.fn();
  render(
    <ReviewEditor
      draft={draft}
      onChange={onChange}
      canAccept
      onSave={vi.fn()}
      onRefresh={vi.fn()}
    />,
  );
  const accept = screen.getByRole("option", {
    name: "Принять предложенное значение",
  }) as HTMLOptionElement;
  expect(accept.disabled).toBe(true);
  fireEvent.change(screen.getByLabelText("Наличие ошибки"), {
    target: { value: "no_error" },
  });
  expect(onChange.mock.calls[0][0].assessment).toBe("no_error");
});
it("409 сохраняет ввод и требует сверки перед повтором", async () => {
  const draft: Draft = {
    assessment: "confirmed_error",
    decision: "accepted",
    comment: "Мой комментарий",
    suggestionId: "s",
  };
  const save = vi.fn().mockRejectedValue(new ApiError(409, "version_conflict"));
  render(
    <ReviewEditor
      draft={draft}
      onChange={vi.fn()}
      canAccept
      onSave={save}
      onRefresh={vi.fn().mockResolvedValue(undefined)}
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Сохранить решение" }));
  await screen.findByText(/Версия рассмотрения изменилась/);
  expect(
    (screen.getByLabelText("Комментарий специалиста") as HTMLTextAreaElement)
      .value,
  ).toBe("Мой комментарий");
  expect(
    (
      screen.getByRole("button", {
        name: "Сохранить решение",
      }) as HTMLButtonElement
    ).disabled,
  ).toBe(true);
  fireEvent.click(
    screen.getByRole("button", { name: "Сверить свежую версию" }),
  );
  await waitFor(() =>
    expect(
      (
        screen.getByRole("button", {
          name: "Сохранить решение",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(false),
  );
  expect(save).toHaveBeenCalledTimes(1);
});
